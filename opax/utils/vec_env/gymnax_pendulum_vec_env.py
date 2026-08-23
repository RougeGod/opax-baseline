import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnax.environments.classic_control.pendulum import angle_normalize
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

    def __init__(self, n_envs: int, seed: int, time_limit: int = 200):
        self.num_envs = n_envs
        self.env, self.env_params = gymnax.make("Pendulum-v1")
        self.env_params = self.env_params.replace(max_steps_in_episode=time_limit)
        self.time_limit = time_limit
        self._raw_action_low = -self.env_params.max_torque
        self._raw_action_high = self.env_params.max_torque
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._reset_fn = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        self._step_fn = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0, 0, None)))

        self.rng = jax.random.PRNGKey(seed)
        self._state = None
        self.reset(seed=seed)

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
    def __init__(self, target_angle, seed):
        super(CompiledPendulumTargetAngleEnv, self).__init__(n_envs = 1, seed=seed)
        
        #no control costs on these envs for comparisons with MUE (vanilla envs!)
        self._reward_model = PendulumReward(action_space = self.env.action_space, ctrl_cost_weight=0.00, 
                                            target_angle=target_angle, sparse=False)
                                            
    #impure step function which modifies env state and returns a gymnasium-style     
    #next_obs, reward, terminate, truncate, info tuple
    def step(self, action):
        next_state, reward = self.pure_step(self.state, action)
        self.state = next_state
        next_obs = self._get_obs(self.state)
        return next_obs, reward, False, False, {}
        
        
            
    #pure step function, accepting state and action, returning new_state.
    #step() handles everything else
    @jax.jit(static_argnames=["self"])    
    def pure_step(self, state, action):
        
        reward = self._reward_model.predict(state, action)
        action = jnp.clip(action, self.min_torque, self.max_torque)
        #constants: can be read during tracing/execution
        g, m, l, dt= self.g, self.m, self.l, self.dt
        theta, theta_dot = state
        theta_doubledot = (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u)
        theta_dot += theta_doubledot * dt
        theta += theta_dot * dt
        next_state = jnp.array([theta, theta_dot])
        
        return next_state, reward, False

        
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
