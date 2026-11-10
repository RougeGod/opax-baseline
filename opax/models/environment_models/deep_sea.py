from opax.models.reward_model import RewardModel
import jax
import jax.numpy as jnp
from functools import partial


class DeepSeaRewardModel(RewardModel):
    """Reward of gymnax's deterministic DeepSea (bsuite) on CompiledDeepSeaVecEnv observations,
    the normalized position (row, column) / (size - 1).

    gymnax pays +1 for moving right from the last column (only reachable in the last row)
    and charges unscaled_move_cost / size for every move right. The reward depends on the
    pre-step observation and the action only, as in gymnax's step_reward.

    The column is rounded to the nearest cell, so the reward is exact on real observations
    and tolerates the small errors of the model's continuous predictions.
    Actions are continuous in [-1, 1]; a > 0 means "right", as in CompiledDeepSeaVecEnv.
    """

    def __init__(self, size: int, unscaled_move_cost: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = size
        self.unscaled_move_cost = unscaled_move_cost

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, obs, action, next_obs=None, rng=None):
        column = jnp.round(obs[..., 1] * (self.size - 1))
        in_last_column = (column == self.size - 1).astype(obs.dtype)
        right = (action[..., 0] > 0).astype(obs.dtype)
        reward = right * in_last_column - right * self.unscaled_move_cost / self.size
        return reward.reshape(-1).squeeze()
