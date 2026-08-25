import jax.numpy as jnp
import jax.random

from opax.utils.replay_buffer import ReplayBuffer
import numpy as np
from typing import Optional


class DummyAgent(object):

    def __init__(self, train_steps: int = 1, batch_size: int = 256, num_epochs: int = -1, max_train_steps: int = 5000):
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_train_steps = max_train_steps
        self.best_sequences = None
        self.act_in_train = lambda obs, rng, best_sequences: self.act_in_jax(obs, rng, best_sequences, eval=False)
        pass

    def act(self, obs: np.ndarray, rng=None, eval: bool = False, eval_idx: int = 0):
        # Eager, non-traced convenience wrapper (used by the old, plain-Python
        # eval_policy loop) — reads self.best_sequences as a default and
        # discards whatever updated state comes back, since a single eager
        # call isn't part of any warm-started sequence of steps.
        action, _ = self.act_in_jax(jnp.asarray(obs), rng, self.best_sequences, eval=eval, eval_idx=eval_idx)
        return np.asarray(action)

    def act_in_jax(self, obs: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None, eval: bool = False, eval_idx: int = 0):
        NotImplementedError

    def train_step(self,
                   rng,
                   buffer: ReplayBuffer,
                   validate: bool = True,
                   log_results: bool = True,
                   ):
        NotImplementedError

    def prepare_agent_for_rollout(self):
        pass
