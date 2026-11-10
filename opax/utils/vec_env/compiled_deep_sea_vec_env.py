import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from gymnax.environments.bsuite.deep_sea import DeepSea


class CompiledDeepSeaVecEnv:
    """Vectorized, jittable wrapper around gymnax's DeepSea (bsuite), following
    ``CompiledMountainCarVecEnv``/``GymnaxPendulumVecEnv``.

    Exposes the surface ``DummyTrainer``/``ModelBasedTrainer`` rely on (``num_envs``,
    ``observation_space``, ``action_space``, ``time_limit``, ``_state``,
    ``pure_reset``/``pure_step``, ``reset``/``step``, ``sample_obs``, ``applied_action``).

    Uses gymnax's deterministic DeepSea dynamics and reward unchanged. Two things are adapted
    so the continuous-action, vector-observation agent can use it:
    - actions are ``Box(-1, 1, (1,))``; ``a > 0`` is "right" (discrete action 1), else "left".
      ``applied_action`` maps a chosen action to the one the env actually applies (+1 or -1),
      and the trainer stores that in the transitions;
    - observations are the ``size x size`` one-hot grid flattened to ``size**2``.

    Every episode lasts exactly ``size`` steps (one row per step), so ``time_limit = size``.
    gymnax auto-resets at the end of an episode, so ``next_obs`` at ``done`` is the reset obs,
    which the trainer's ``mask_out_resets`` relies on.

    No ``env_name`` is set, so the trainer skips ``get_dynamics_mae`` (no ground truth for it).
    """

    def __init__(self, n_envs: int, seed: int, size: int = 50):
        self.num_envs = n_envs
        self.size = size
        self.time_limit = size
        self.env = DeepSea(size=size)
        self.env_params = self.env.default_params.replace(max_steps_in_episode=size)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._reset_fn = jax.vmap(self.env.reset, in_axes=(0, None))
        self._step_fn = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))
        # valid (row, column) cells: the column moves at most one step right per row
        rows, cols = np.tril_indices(size)
        self._valid_cells = jnp.asarray(rows * size + cols)

        self.rng = jax.random.PRNGKey(seed)
        self._state = None
        self.reset(seed=seed)

    @staticmethod
    def applied_action(action):
        """The action the env actually applies for a chosen action in [-1, 1]:
        +1 (right) if a > 0, else -1 (left). Elementwise, so any batch shape works."""
        return jnp.where(jnp.asarray(action) > 0, 1.0, -1.0).astype(jnp.float32)

    @jax.jit(static_argnames=["self", "num_samples"])
    def sample_obs(self, rng: jax.Array, num_samples: int):
        #num_samples uniformly sampled reachable cells
        cells = jax.random.randint(rng, (num_samples, self.observation_space.shape[0]), 
                                   minval=0, maxval=self.size).astype(jnp.float32)
        return cells

    def _to_discrete(self, action: jnp.ndarray) -> jnp.ndarray:
        action = jnp.asarray(action).reshape(self.num_envs, -1)
        return (action[:, 0] > 0).astype(jnp.int32)

    @jax.jit(static_argnames=["self"])
    def pure_reset(self, rng: jax.Array):
        """Pure, jittable reset: no ``self`` mutation, no NumPy conversion.

        :param rng: jax.Array, single rng key for this reset (split internally across ``num_envs``).
        :return: obs, env_state
        """
        reset_keys = jax.random.split(rng, self.num_envs)
        _, env_state = self._reset_fn(reset_keys, self.env_params)
        obs = jnp.stack([env_state.row, env_state.column], axis=-1).astype(jnp.float32) / (self.size - 1)
        return obs, env_state

    def reset(self, seed=None):
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, reset_rng = jax.random.split(self.rng)
        obs, self._state = self.pure_reset(reset_rng)
        return np.asarray(obs), [{} for _ in range(self.num_envs)]

    @jax.jit(static_argnames=["self"])
    def pure_step(self, rng: jax.Array, env_state, action: jnp.ndarray):
        """Pure, jittable single-call step: no ``self`` mutation, no NumPy conversion.

        :param rng: jax.Array, single rng key for this step (split internally across ``num_envs``).
        :param env_state: gymnax DeepSea EnvState pytree, batched over ``num_envs``.
        :param action: array of num_envs actions in [-1, 1], e.g. shaped (num_envs, 1).
        :return: next_obs, next_state, reward, done
        """
        step_keys = jax.random.split(rng, self.num_envs)
        next_obs, next_state, reward, done, _ = self._step_fn(
            step_keys, env_state, self._to_discrete(action), self.env_params
        )
        next_obs = jnp.stack([next_state.row, next_state.column], axis=-1).astype(jnp.float32) / (self.size - 1)
        #make observation from state so as not to store a massive one-hot matrix
        #division is to normalize the obs between 0 and 1, breaks if size == 1
        return next_obs, next_state, reward, done

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
