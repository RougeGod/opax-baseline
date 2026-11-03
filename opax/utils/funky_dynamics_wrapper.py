from gymnax.wrappers.purerl import GymnaxWrapper
from gymnax.environments import spaces, environment
import jax
import jax.numpy as n

class FunkyDynamicsWrapper(GymnaxWrapper):

    def action_space(self, params) -> spaces.Dict:
        return spaces.Dict({"env_action": self._env.action_space(params), "noise": spaces.Discrete(2)})

    @jax.jit(static_argnames=["self"])
    def step_env(self, rng_key: jax.Array, state: environment.EnvState,
             action: spaces.Dict, params: environment.EnvParams | None = None):
        next_obs, next_state, reward, done, info = jax.lax.cond(action["noise"], 
                                                   self.funky_dynamics, #true_fn 
                                                   self._env.step_env, #false_fn
                                                   rng_key, state, action["env_action"], params) #arguments to function
        return next_obs, next_state, reward, done, info

    @jax.jit(static_argnames=["self"])
    def funky_dynamics(self, rng_key: jax.Array, env_state, action, params):
        next_obs = random_obs(rng_key, params)
        next_state = env_state #freeze the environmental state
        next_state.time = env_state.time + 1
        reward = 0
        done = False
        info = {"discount": self.discount(next_state, params)}
        return next_obs, next_state, reward, done, info
        

    @jax.jit(static_argnames=["self"])
    def random_obs(self, rng_key, params, clip_value=1e6):
        rng_key, subkey = jax.random.split(rng_key)
        obs_dim = self._env.observation_space(params).shape
        high = n.clip(self._env.observation_space(params).high, -clip_value, clip_value)
        low = n.clip(self._env.observation_space(params).low, -clip_value, clip_value)
        return jax.random.uniform(rng_key, n.shape(obs_dim), minval=low, maxval=high)
