import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnax.environments.classic_control.pendulum import angle_normalize, EnvState
from opax.models.environment_models.pendulum_swing_up import PendulumReward

    

class GymnaxPendulumVecEnv:
    """Vectorized pendulum environment stepped with gymnax instead of gymnasium.

    Drop-in replacement for the ``make_vec_env(CustomPendulumEnv, ...)`` training
    env: exposes the same surface ``DummyTrainer``/``ModelBasedTrainer`` rely on
    (``num_envs``, ``observation_space``, ``action_space``, ``step``, ``reset``),
    but steps all ``n_envs`` copies as a single jitted ``jax.vmap`` call over
    gymnax's pure-JAX pendulum dynamics instead of dispatching to ``n_envs``
    separate gymnasium/NumPy environments (one per OS process/loop iteration).

    Uses gymnax's standard Pendulum-v1 dynamics and reward (vanilla swing-up-to
    -upright task) rather than the repo's ``CustomPendulumEnv``/``PendulumReward``
    target-angle machinery. The action space is exposed as ``Box(-1, 1)`` to
    match what the gymnasium ``RescaleAction`` wrapper produced for the env this
    replaces, so the agent/optimizer code that assumes actions in [-1, 1] is
    unaffected; actions are rescaled to the raw torque range internally before
    being handed to gymnax.
    """
    env_name = "Pendulum-v1"

    def __init__(self, n_envs: int, seed: int, time_limit: int = 200, reset_always_down=False):
        self.num_envs = n_envs
        self.env, self.env_params = gymnax.make("Pendulum-v1")
        self.env_params = self.env_params.replace(max_steps_in_episode=time_limit)
        self.time_limit = time_limit
        self._raw_action_low = -self.env_params.max_torque
        self._raw_action_high = self.env_params.max_torque
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset_always_down = reset_always_down
        self._reset_fn = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        self._step_fn = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0, 0, None)))

        self.rng = jax.random.PRNGKey(seed)
        self._state = None
        self.reset(seed=seed)

    @jax.jit(static_argnames=["self", "num_samples"])
    def sample_obs(self, rng: jax.Array, num_samples: int):
        """Instead of sampling states from Box, make sure the states 
        are valid pendulum states.

        :param rng: jax.Array, rng key
        :param num_samples: int, number of observations to sample
        :return: observations, shape (num_samples, 3)
        """
        high = jnp.array([jnp.pi, 1.0])
        samples = jax.random.uniform(rng, (num_samples, 2), minval=-high, maxval=high)
        theta, thetadot = samples[:, 0], samples[:, 1]
        return jnp.stack([jnp.cos(theta), jnp.sin(theta), thetadot], axis=-1).astype(jnp.float32)

    def _rescale_action(self, action: np.ndarray) -> jnp.ndarray:
        action = jnp.clip(jnp.asarray(action), -1.0, 1.0)
        return self._raw_action_low + (action + 1.0) * 0.5 * (self._raw_action_high - self._raw_action_low)

    @jax.jit(static_argnames=["self"])
    def pure_reset(self, rng: jax.Array):
        """Pure, jittable reset: no ``self`` mutation, no NumPy conversion.
        Safe to use inside ``jax.lax.scan``/``jax.jit``/``jax.vmap`` (e.g. from
        ``dummy_trainer.better_eval_policy``), unlike ``reset`` below which
        owns/mutates ``self.rng``/``self._state`` and crosses back to NumPy.

        :param rng: jax.Array, single rng key for this reset (split internally
            across ``num_envs``), mirroring how ``pure_step`` derives its own
            keys.
        :return: obs, env_state
        """
        reset_keys = jax.random.split(rng, self.num_envs)
        obs, env_state = self._reset_fn(reset_keys, self.env_params)
        return obs, env_state

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, reset_rng = jax.random.split(self.rng)
        obs, self._state = self.pure_reset(reset_rng)
        obs_np = np.asarray(obs)
        return obs_np, [{} for _ in range(self.num_envs)]
    
    @jax.jit(static_argnames=["self"])
    def straight_down_reset(self, seed=None, env_params=None):
        #reset is not random and does not depend on env params
        #but we need it to have the same signature. 
        #ignore these inputs
        reset_state = EnvState(time=jnp.zeros((), dtype=jnp.float32),
                               theta=jnp.zeros((), dtype=jnp.float32),
                               theta_dot=jnp.zeros((), dtype=jnp.float32),
                               last_u=jnp.zeros((), dtype=jnp.float32))
                                
        return self.env.get_obs(reset_state), reset_state

    @jax.jit(static_argnames=["self"])
    def pure_step(self, rng: jax.Array, env_state, action: jnp.ndarray):
        """Pure, jittable single-call step: no ``self`` mutation, no NumPy
        conversion. Safe to use inside ``jax.lax.scan``/``jax.jit`` (e.g. from
        ``dummy_trainer.step_env``), unlike ``step`` below which owns/mutates
        ``self.rng``/``self._state`` and crosses back to NumPy.

        :param rng: jax.Array, single rng key for this step (split internally
            across ``num_envs`), mirroring how ``step`` derives its own keys.
        :param env_state: gymnax EnvState pytree, batched over ``num_envs``.
        :param action: array shaped (num_envs, *action_shape), in [-1, 1].
        :return: next_obs, next_state, reward, done
        """
        scaled_action = self._rescale_action(action)
        step_keys = jax.random.split(rng, self.num_envs)
        next_obs, next_state, reward, done, _ = self._step_fn(
            step_keys, env_state, scaled_action, self.env_params
        )
        return next_obs, next_state, reward, done

    @staticmethod
    @jax.jit
    def _get_obs(state):
        theta, thetadot = state.theta, state.theta_dot
        return jnp.stack([jnp.cos(theta), jnp.sin(theta), thetadot], axis=-1).astype(jnp.float32)


    def step(self, action: np.ndarray):
        self.rng, step_rng = jax.random.split(self.rng)
        next_obs, next_state, reward, done = self.pure_step(step_rng, self._state, action)
        self._state = next_state

        next_obs_np = np.asarray(next_obs)
        reward_np = np.asarray(reward)
        done_np = np.asarray(done)
        # gymnax's step already auto-resets internally when done (lax.select
        # between the stepped and freshly-reset state/obs), matching the
        # gymnasium VecEnv auto-reset convention this codebase relies on, so
        # next_obs_np is already the right thing to feed to the next step.
        info = [
            {"current_env_state": next_obs_np[i], "last_done": done_np[i]}
            for i in range(self.num_envs)
        ]
        return next_obs_np, reward_np, done_np, done_np, info
     
class CompiledPendulumTargetAngleEnv(GymnaxPendulumVecEnv):
    """Vectorized pendulum with a target-angle reward (a generalization of
    ``GymnaxPendulumSwingUp`` for arbitrary target angles, e.g.
    ``target_angle=0.0`` for swing-up, ``target_angle=pi`` for keep-down).

    Dynamics are entirely inherited from ``GymnaxPendulumVecEnv``
    (``super().pure_step``/``reset``) — gymnax's real, tested Pendulum-v1
    physics, including action rescaling and max-speed clipping — only the
    reward is replaced with ``PendulumReward``'s target-angle formula. This
    class never hand-rolls the physics itself, so there is only one place
    (the inherited dynamics) that can get the state representation wrong.

    Convention used throughout this file: ``state``/``env_state`` is always
    a gymnax ``EnvState`` pytree (attribute access: ``.theta``/
    ``.theta_dot``); ``obs`` is always the flattened
    ``[cos theta, sin theta, theta_dot]`` array that ``PendulumReward`` and
    the replay buffer expect. The only place the two are converted is
    ``_get_obs`` (inherited from ``GymnaxPendulumVecEnv``) — nothing here
    reads ``.theta``/``.theta_dot`` off an obs array or indexes into an
    ``EnvState`` positionally.
    """

    def __init__(self, target_angle, seed):
        super(CompiledPendulumTargetAngleEnv, self).__init__(n_envs=1, seed=seed)

        #no control costs on these envs for comparisons with MUE (vanilla envs!)
        self._reward_model = PendulumReward(action_space=self.action_space, ctrl_cost_weight=0.00,
                                            target_angle=target_angle, sparse=False)

    def pure_step(self, rng: jax.Array, env_state, action: jnp.ndarray):
        """Pure, jittable step. Reward is computed on the pre-transition obs
        and action (PendulumReward's r(s, a) convention, matching
        CustomPendulumEnv.step()) before delegating the actual dynamics to
        the inherited gymnax step, so ``next_state`` stays a proper
        ``EnvState`` (never a bare array) and can be fed straight back into
        the next call.
        """
        obs = self._get_obs(env_state)
        reward = self._reward_model.predict(obs, action)
        next_obs, next_state, _vanilla_reward, done = super().pure_step(rng, env_state, action)
        return next_obs, next_state, reward, done

        
class GymnaxPendulumSwingUp(GymnaxPendulumVecEnv):
    def __init__(self, num_envs: int, seed: int, time_limit: int=200):
        super(GymnaxPendulumSwingUp, self).__init__(num_envs, seed, time_limit)
        
    def pure_step(self, rng: jax.Array, env_state, action: jax.Array):
        next_obs, next_state, vanilla_reward, done = super().pure_step(rng, env_state, action)
        reward = self.state_reward(env_state, target_angle=0.0)
        return next_obs, next_state, reward, done
        
    @staticmethod
    @jax.jit
    def state_reward(state, target_angle: float = 0.0):
        """Compute reward associated with state dynamics."""
        theta = angle_normalize(state.theta - target_angle)
        return -(theta ** 2 + 0.1 * state.theta_dot ** 2)
