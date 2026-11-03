import jax
import jax.numpy as jnp
import gymnax
from gymnax.environments import spaces
from opax.models.environment_models.mountain_car import MountainCarDynamics


_DUMMY_KEY = jax.random.key(0)
_MOUNTAIN_CAR_DYNAMICS = MountainCarDynamics()

def make_state(env_name: str, s1, s2):
    if env_name == "Pendulum-v1":
        from gymnax.environments.classic_control.pendulum import EnvState

        return EnvState(theta=s1, theta_dot=s2, last_u=jnp.zeros(()), time=0)
    if env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        if env_name == "MountainCar-v0":
            from gymnax.environments.classic_control.mountain_car import EnvState
        else:
            from gymnax.environments.classic_control.continuous_mountain_car import (
                EnvState,
            )

        return EnvState(position=s1, velocity=s2, time=0)
    if env_name == "CartPole-v1":
        from gymnax.environments.classic_control.cartpole import EnvState

        return EnvState(
            x=s1,
            x_dot=jnp.zeros_like(s1),
            theta=s2,
            theta_dot=jnp.zeros_like(s2),
            time=0,
        )
    if env_name == "Acrobot-v1":
        from gymnax.environments.classic_control.acrobot import EnvState

        return EnvState(
            joint_angle1=s1,
            joint_angle2=s2,
            velocity_1=jnp.zeros_like(s1),
            velocity_2=jnp.zeros_like(s2),
            time=0,
        )
    raise ValueError(env_name)

def true_transition(env, params, env_name, s1, s2, action):
    """Ground-truth (delta_obs, reward, terminated) for a single transition."""
    if env_name == "MountainCarContinuous-v0":
        # The training env (CustomMountainCar / CompiledMountainCarVecEnv) steps
        # MountainCarDynamics, not gymnax's physics, which terminate at the goal
        # and zero the velocity there instead of at the left wall. Use the same
        # function so the ground truth matches the env the data comes from.
        obs = obs_from_generic_coords(s1, s2, env_name)
        next_obs, reward = _MOUNTAIN_CAR_DYNAMICS.evaluate(obs=obs, action=jnp.atleast_1d(action))
        # CustomMountainCar never terminates, the episode only ends at the time limit
        return next_obs - obs, reward, jnp.zeros((), dtype=jnp.float32)
    state = make_state(env_name, s1, s2)
    obs = env.get_obs(state)
    if isinstance(env.action_space(params), spaces.Discrete):
        action = jnp.asarray(action, dtype=jnp.int32)
    next_obs, _, reward, terminated, info = env.step(_DUMMY_KEY, state, action, params)
    return next_obs - obs, reward, terminated.astype(jnp.float32)


def true_delta_obs(env, params, env_name, s1, s2, action):
    """Batched ground-truth observation deltas, shape (N, obs_dim)."""
    delta_obs, _, _ = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return delta_obs


def true_reward(env, params, env_name, s1, s2, action):
    """Batched ground-truth rewards, shape (N,)."""
    _, reward, _ = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return reward


def true_terminated(env, params, env_name, s1, s2, action):
    """Batched ground-truth termination flags, shape (N,)."""
    _, _, terminated = jax.vmap(
        lambda x1, x2, a: true_transition(env, params, env_name, x1, x2, a)
    )(s1, s2, action)
    return terminated


def generate_validation_data(env, env_params, env_name, num_val=1000, batch=None):
    val_rng = jax.random.PRNGKey(42)
    
    if batch is None:
        val_rng, val_obs, val_s1, val_s2, val_act = sample_validation_batch(
            val_rng, env, env_params, env_name, num_val
        )
    else:
        val_rng, val_obs, val_s1, val_s2, val_act = batch

    val_true_delta_obs = true_delta_obs(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )
    val_true_reward = true_reward(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )
    val_true_terminated = true_terminated(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )

    return val_obs, val_act, val_true_delta_obs, val_true_reward, val_true_terminated

def obs_from_generic_coords(s1, s2, env_name):
    match env_name:
        case "Pendulum-v1":
            #s1 is theta but the observation is x, y, dtheta/dt
            return jnp.stack([jnp.cos(s1), jnp.sin(s1), s2], axis=-1)
        case "MountainCarContinuous-v0":
            #s1 is position, s2 is velocity, observed directly
            return jnp.stack([s1, s2], axis=-1)
        case _:
            raise NotImplementedError(f"todo: support other environments like {env_name}")
            

def sample_validation_batch(rng, env, env_params, env_name: str, num_samples: int):

    #n_act = num_actions(env, env_params) #not necessary for some envs incl. pendulum
    
    if env_name == "Pendulum-v1":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        #4 keys: rng creates the next key, s1 for angle, s2 for 
        #angular velocity, and key_act for action
        s1 = jax.random.uniform(key_s1, (num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.max_speed,
            maxval=env_params.max_speed,
        )
        actions = jax.random.uniform(
            key_act,
            (num_samples, 1),
            minval=-env_params.max_torque,
            maxval=env_params.max_torque,
        )
    elif env_name == "MountainCarContinuous-v0":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        #s1 for position, s2 for velocity, uniform over the observation space
        #like CustomMountainCar.sample_obs
        s1 = jax.random.uniform(
            key_s1,
            (num_samples,),
            minval=env_params.min_position,
            maxval=env_params.max_position,
        )
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.max_speed,
            maxval=env_params.max_speed,
        )
        actions = jax.random.uniform(
            key_act,
            (num_samples, 1),
            minval=env_params.min_action,
            maxval=env_params.max_action,
        )
    else: raise NotImplementedError
    obs = obs_from_generic_coords(s1, s2, env_name)
    
    return rng, obs, s1, s2, actions
        
