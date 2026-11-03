import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnax.environments.classic_control.continuous_mountain_car import EnvState
from opax.models.environment_models.mountain_car import MountainCarDynamics


class CompiledMountainCarVecEnv:
    """Vectorized, jittable drop-in replacement for the
    ``make_vec_env(CustomMountainCar, wrapper_class=RescaleAction(TimeLimit(...)))`` env.

    Exposes the same surface ``DummyTrainer``/``ModelBasedTrainer`` rely on
    (``num_envs``, ``observation_space``, ``action_space``, ``time_limit``,
    ``_state``, ``pure_reset``/``pure_step``, ``reset``/``step``), following
    ``GymnaxPendulumVecEnv``.

    The transition is ``MountainCarDynamics.evaluate``, the same function
    ``CustomMountainCar.step`` calls, rather than gymnax's
    ``MountainCarContinuous-v0`` physics, which differ from it: gymnax ends the
    episode when the goal is reached and zeroes the velocity at the goal
    instead of at the left wall. Like ``CustomMountainCar``, reaching the goal
    never terminates the episode; it only ends at ``time_limit``, where the env
    auto-resets (``next_obs`` is the fresh reset obs), matching the gymnax
    auto-reset convention the trainer's reset masking relies on.

    The action space is ``Box(-1, 1)`` and actions are passed through
    unchanged: ``CustomMountainCar``'s own action bounds are already [-1, 1],
    so its ``RescaleAction(-1, 1)`` wrapper was the identity.
    """
    env_name = "MountainCarContinuous-v0"

    def __init__(self, n_envs: int, seed: int, time_limit: int = 200):
        self.num_envs = n_envs
        self.time_limit = time_limit
        self.dynamics_model = MountainCarDynamics()
        car = self.dynamics_model.env
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([car.min_position, -car.max_speed], dtype=np.float32),
            high=np.array([car.max_position, car.max_speed], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_fn = jax.vmap(self._reset_single)
        self._step_fn = jax.vmap(self._step_single)

        self.rng = jax.random.PRNGKey(seed)
        self._state = None
        self.reset(seed=seed)

    @staticmethod
    def _get_obs(state: EnvState) -> jnp.ndarray:
        return jnp.stack([state.position, state.velocity], axis=-1).astype(jnp.float32)

    def _reset_single(self, key: jax.Array):
        # same initial distribution as CustomMountainCar.reset
        position = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=position, velocity=jnp.array(0.0), time=0)
        return self._get_obs(state), state

    def _step_single(self, key: jax.Array, state: EnvState, action: jnp.ndarray):
        next_obs, reward = self.dynamics_model.evaluate(obs=self._get_obs(state), action=action)
        stepped_state = EnvState(position=next_obs[0], velocity=next_obs[1], time=state.time + 1)
        done = stepped_state.time >= self.time_limit
        # auto-reset at the time limit, as gymnax's step does
        reset_obs, reset_state = self._reset_single(key)
        next_state = jax.tree_util.tree_map(lambda r, s: jax.lax.select(done, r, s), reset_state, stepped_state)
        next_obs = jax.lax.select(done, reset_obs, self._get_obs(stepped_state))
        return next_obs, next_state, reward, done

    @jax.jit(static_argnames=["self"])
    def pure_reset(self, rng: jax.Array):
        """Pure, jittable reset: no ``self`` mutation, no NumPy conversion.

        :param rng: jax.Array, single rng key for this reset (split internally across ``num_envs``).
        :return: obs, env_state
        """
        reset_keys = jax.random.split(rng, self.num_envs)
        return self._reset_fn(reset_keys)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, reset_rng = jax.random.split(self.rng)
        obs, self._state = self.pure_reset(reset_rng)
        obs_np = np.asarray(obs)
        return obs_np, [{} for _ in range(self.num_envs)]

    @jax.jit(static_argnames=["self"])
    def pure_step(self, rng: jax.Array, env_state, action: jnp.ndarray):
        """Pure, jittable single-call step: no ``self`` mutation, no NumPy conversion.

        :param rng: jax.Array, single rng key for this step (split internally across ``num_envs``),
            only used to sample the reset state of envs that hit the time limit.
        :param env_state: EnvState pytree, batched over ``num_envs``.
        :param action: array of num_envs actions in [-1, 1], e.g. shaped (num_envs, 1) or (num_envs,).
        :return: next_obs, next_state, reward, done
        """
        # one (1,) action per env: MountainCarRewardModel indexes action[..., 0]
        action = jnp.asarray(action).reshape(self.num_envs, -1)
        step_keys = jax.random.split(rng, self.num_envs)
        return self._step_fn(step_keys, env_state, action)

    def step(self, action: np.ndarray):
        self.rng, step_rng = jax.random.split(self.rng)
        next_obs, next_state, reward, done = self.pure_step(step_rng, self._state, action)
        self._state = next_state

        next_obs_np = np.asarray(next_obs)
        reward_np = np.asarray(reward)
        done_np = np.asarray(done)
        info = [
            {"current_env_state": next_obs_np[i], "last_done": done_np[i]}
            for i in range(self.num_envs)
        ]
        return next_obs_np, reward_np, done_np, done_np, info
