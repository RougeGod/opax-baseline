import jax
from opax.utils.replay_buffer import Transition, BufferState
from opax.utils.replay_buffer import JaxReplayBuffer as ReplayBuffer
from opax.agents.dummy_agent import DummyAgent
from typing import Callable, Optional, Union
from opax.utils.csv_logger import CSVLogger
from copy import deepcopy
import numpy as np
import jax.numpy as jnp
from opax.utils.vec_env import VecEnv
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
from gymnasium import Env
import orbax.checkpoint as ocp
from datetime import datetime
import os


# TODO learn deltas, validate_agent should only be used by model based agent.
class DummyTrainer(object):
    """Dummy Trainer for training a RL agent

    env: VecEnv, Vectorized environment for RL
    agent: DummyAgent, RL Agent
    buffer_size: int, Replay buffer size
    total_train_steps: int, Total number of training steps for the RL agent.
    train_freq: int, frequency of training wrt rollouts in the real environment
    eval_freq: int, frequency of evaluation wrt rollouts in the real environment.
    seed: int
    exploration_steps: int, number of random data points collected before agent update.
    rollout_steps: int, Number of steps per rollout in the environment
    eval_episodes: int, Number of episodes used for evaluation
    agent_name: str, Name of the agent
    use_wandb: bool, boolean to indicate whether metrics should be logged to csv files
    validate: bool, boolean to indicate policy validation (generally only used for model based RL)
    normalize: bool, boolean to indicate if data is normalized or not
    agent_save_loc: str, location to save the agent checkpoint
    agent_load_loc: str, location to load the agent checkpoint from
    action_normalize: bool, boolean to indicate if actions should be normalized
    learn_deltas: bool, boolean to indicate if model learns a delta
    record_test_video: bool, boolean to indicate if video should be recorded for model evaluation
    video_prefix: str, prefix of the recorded video
    video_folder: str, folder where the videos are stored
    test_env: Optional[Union[VecEnv, Env]], environment on which the agent is tested.
    """

    def __init__(self,
                 env: VecEnv,
                 agent: DummyAgent,
                 buffer_size: int = int(1e6),
                 total_train_steps: int = int(1e6),
                 train_freq: int = 100,
                 eval_freq: int = 1000,
                 seed: int = 0,
                 exploration_steps: int = int(1e4),
                 rollout_steps: int = 200,
                 eval_episodes: int = 100,
                 agent_name: str = "DummyAgent",
                 use_wandb: bool = True,
                 validate: bool = False,
                 normalize: bool = False,
                 agent_save_loc: str = None,
                 agent_load_loc: str = None,
                 action_normalize: bool = False,
                 learn_deltas: bool = True,
                 record_test_video: bool = False,
                 video_prefix: str = "",
                 video_folder: str = "./opax_results",
                 test_env: Optional[Union[VecEnv, Env]] = None,
                 ):
        self.env = env
        self.num_envs = max(env.num_envs, 1)
        self.buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            max_size=buffer_size,
            normalize=normalize,
            action_normalize=action_normalize,
            learn_deltas=learn_deltas
        )
        self.buffer_size = buffer_size
        
        self.buffer_state = self.buffer.initialize_buffer_state()

        self.eval_episodes = eval_episodes
        self.agent_name = agent_name
        self.rng = jax.random.PRNGKey(seed)
        self.use_wandb = use_wandb
        self.validate = validate
        self.agent_save_loc = agent_save_loc
        self.agent_load_loc = agent_load_loc
        self.total_train_steps = total_train_steps
        self.train_freq = train_freq
        self.eval_freq = eval_freq
        self.exploration_steps = exploration_steps
        self.rollout_steps = rollout_steps
        self.agent = agent
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.video_dir_name = video_folder + 'video' + str(seed) + video_prefix + dt_string
        self.record_test_video = record_test_video
        self.logger = None
        if self.use_wandb:
            log_dir = video_folder + 'csv_logs' + str(seed) + video_prefix + dt_string
            self.logger = CSVLogger(log_dir)

        if self.agent_load_loc is not None:
            self.load_agent(self.agent_load_loc)

        # wraps test environment with recordvideo wraper if record_test_video is true
        def get_test_env_wrapper(env_num):
            video_dir_name = self.video_dir_name
            if self.record_test_video:
                video_dir_name = self.video_dir_name + '/env_id_' + str(env_num)
                test_env_wrapper = lambda x: RecordVideo(x, video_folder=video_dir_name,
                                                         episode_trigger=lambda x: True)
            else:
                test_env_wrapper = lambda x: x
            return test_env_wrapper, video_dir_name

        # Loop over test envs and store them.
        if test_env is None:
            test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=0)
            self.test_env = [
                test_env_wrapper(deepcopy(self.env.envs[0]))
            ]
            self.video_dirs = [video_dir_name]
        else:
            self.test_env = []
            self.video_dirs = []
            if isinstance(test_env, list):
                for env_id, env in enumerate(test_env):
                    test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=env_id)
                    self.video_dirs.append(video_dir_name)
                    if isinstance(env, VecEnv):
                        for e in env.envs:
                            self.test_env.append(
                                test_env_wrapper(e)
                            )
                    else:
                        self.test_env.append(
                            test_env_wrapper(env)
                        )
            else:
                test_env_wrapper, video_dir_name = get_test_env_wrapper(env_num=0)
                self.video_dirs = [video_dir_name]
                if isinstance(test_env, VecEnv):
                    for e in test_env.envs:
                        self.test_env.append(
                            test_env_wrapper(e)
                        )
                else:
                    self.test_env.append(
                        test_env_wrapper(test_env)
                    )
        assert len(self.test_env) == self.num_reward_models, "number of reward models must be the same as number of " \
                                                             "envs"

    def train(self):
        pass

    @staticmethod
    def _is_pure(env) -> bool:
        """True for JAX-native envs (pure_step/pure_reset/_state), which are stepped inside
        jitted scans. Other envs (e.g. dm_control envs built with make_vec_env) have no pure
        step and are stepped in a Python loop on the host, as in the original trainer."""
        return hasattr(env, "pure_step")

    def _to_transitions(self, num_steps, obs_sequence, action_sequence, next_obs_sequence, reward_sequence,
                        dones) -> Transition:
        """Flatten (num_steps, num_envs, ...) rollout sequences into transitions, env by env."""
        num_points = int(num_steps * self.num_envs)
        obs_vec = jnp.reshape(obs_sequence,(num_points,) + self.env.observation_space.shape ,order="F") #concatenate them env by env
        action_vec = jnp.reshape(action_sequence,(num_points,) + self.env.action_space.shape ,order="F")
        reward_vec = jnp.reshape(reward_sequence,(num_points,1), order="F")
        next_obs_vec = jnp.reshape(next_obs_sequence,(num_points,) + self.env.observation_space.shape ,order="F")
        done_vec = jnp.reshape(dones,(num_points,1), order="F")
        return Transition(
            obs=obs_vec,
            action=action_vec,
            reward=reward_vec,
            next_obs=next_obs_vec,
            done=done_vec,
        )

    def _host_rollout(self, obs, policy: Callable, num_steps: int, rng: jax.Array):
        """Python-loop counterpart of the scan in _step_env_scan, for envs without pure_step.

        The policy is still the jitted act_in_jax, so planning runs on device; only the env
        step happens on the host. DummyVecEnv returns the true final obs as next_obs and puts
        the reset obs in info["current_env_state"], so every stored transition is a real one
        and done is the termination flag (not truncation), as in the original trainer.
        :return: (num_steps, num_envs, ...) sequences (obs, action, next_obs, reward, done),
            last_obs, last_done, last policy_state
        """
        model_params, model_props = self.agent.get_model_inputs()
        policy_state = self.agent.policy_state
        # same per-step action keys as the scan path: the first half of each step key
        act_rngs = jax.vmap(jax.random.split)(jax.random.split(rng, num_steps))[:, 0]
        obs_seq, action_seq, next_obs_seq, reward_seq, done_seq = [], [], [], [], []
        last_done = np.zeros(self.num_envs, dtype=bool)
        for act_rng in act_rngs:
            action, policy_state = policy(jnp.asarray(obs), act_rng, policy_state, model_params, model_props)
            action = np.asarray(action).reshape((self.num_envs,) + self.env.action_space.shape)
            next_obs, reward, terminate, truncate, info = self.env.step(action)
            obs_seq.append(np.asarray(obs))
            action_seq.append(action)
            next_obs_seq.append(np.asarray(next_obs))
            reward_seq.append(np.asarray(reward).reshape(-1))
            done_seq.append(np.asarray(terminate).reshape(-1))
            # continue from the reset obs for envs that just finished
            obs = np.stack([np.asarray(x["current_env_state"]) for x in info])
            last_done = np.asarray([x["last_done"] for x in info]).reshape(-1)
        sequences = tuple(np.stack(s) for s in (obs_seq, action_seq, next_obs_seq, reward_seq, done_seq))
        return sequences, obs, last_done, policy_state

    def save_agent(self, step: int = 0, agent_name: Optional[str] = None):
        """Saves the agent PyTree state using Orbax."""
        if self.agent_save_loc is None:
            return

        name = agent_name if agent_name is not None else f"step_{step}"
        save_path = os.path.abspath(os.path.join(self.agent_save_loc, name))
        os.makedirs(self.agent_save_loc, exist_ok=True)

        if hasattr(self.agent, "dynamics_model_list"):
            state = {
                "model_params": [model.model_params for model in self.agent.dynamics_model_list],
                "model_opt_state": [model.model_opt_state for model in self.agent.dynamics_model_list],
                "model_props": [model.model_props for model in self.agent.dynamics_model_list],
                "update_steps": getattr(self.agent, "update_steps", 0),
            }
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(save_path, state, force=True)
        else:
            raise NotImplementedError("save_agent is currently only implemented for ModelBasedAgent.")

    def load_agent(self, agent_directory: str):
        """Loads the agent PyTree state using Orbax."""
        if not os.path.exists(agent_directory):
            raise FileNotFoundError(f"Agent checkpoint directory {agent_directory} does not exist.")

        if hasattr(self.agent, "dynamics_model_list"):
            target_state = {
                "model_params": [model.model_params for model in self.agent.dynamics_model_list],
                "model_opt_state": [model.model_opt_state for model in self.agent.dynamics_model_list],
                "model_props": [model.model_props for model in self.agent.dynamics_model_list],
                "update_steps": getattr(self.agent, "update_steps", 0),
            }
            checkpointer = ocp.PyTreeCheckpointer()
            restored = checkpointer.restore(agent_directory, item=target_state)

            for i, model in enumerate(self.agent.dynamics_model_list):
                alpha = getattr(restored["model_props"][i], "alpha", 1.0)
                model.model_props = restored["model_props"][i] = model.update_model(
                    model_params=restored["model_params"][i],
                    model_opt_state=restored["model_opt_state"][i],
                    alpha=alpha,
                    model_props=restored["model_props"][i]
                )
                
            if "update_steps" in restored:
                self.agent.update_steps = restored["update_steps"]
        else:
            raise NotImplementedError("load_agent is currently only implemented for ModelBasedAgent.")

    

        

    @jax.jit(static_argnames=["self", "policy", "num_steps"])
    def _step_env_scan(self, obs, env_state, policy_state, model_params, model_props, policy: Callable,
                       num_steps: int, rng: jax.Array):
        """Pure scan body pulled out of step_env so it gets a persistent jit
        cache. Called bare inside step_env, jax.lax.scan would otherwise
        retrace and recompile this every single call since there's no
        enclosing jit whose dispatch cache could be reused across calls.
        model_params/model_props are traced inputs (not read off self.agent)
        so each call sees the current model instead of the first-traced one.
        """
        def step_body(carry, step_rng):
            obs, env_state, policy_state = carry
            act_rng, step_rng = jax.random.split(step_rng)
            action, policy_state = policy(obs, act_rng, policy_state, model_params, model_props) #get policy in scope while compiling internal function
            next_obs, next_state, reward, done = self.env.pure_step(step_rng, env_state, action)
            return (next_obs, next_state, policy_state), (obs, action, next_obs, reward, done)

        step_rngs = jax.random.split(rng, num_steps)
        return jax.lax.scan(step_body, (obs, env_state, policy_state), step_rngs)

    def step_env(self, obs: Union[jax.Array, np.ndarray], policy: Callable, num_steps: int,
                 rng: jax.Array) -> [Transition, Union[jax.Array, np.ndarray],
                                                   Union[jax.Array, np.ndarray, bool]]:
        """
        Step in the environment starting at initial state obs. Env is not reset.
        :param obs: Last observation in the environment
        :param policy: Policy function
        :param num_steps: Num steps for rollout
        :param rng: random key for rollout
        :return: transitions, last_obs, last_done
        """
        self.agent.prepare_agent_for_rollout()
        if not self._is_pure(self.env):
            sequences, last_obs, last_done, last_policy_state = self._host_rollout(obs, policy, num_steps, rng)
            self.agent.policy_state = last_policy_state
            return self._to_transitions(num_steps, *sequences), last_obs, last_done

        model_params, model_props = self.agent.get_model_inputs()
        (last_obs, last_state, last_policy_state), (obs_sequence, action_sequence,
                                    next_obs_sequence, reward_sequence,
                                    dones) = self._step_env_scan(
            obs, self.env._state, self.agent.policy_state, model_params, model_props, policy, num_steps, rng)

        self.env._state = last_state #update the env with the result of the pure function
        # unlike better_eval_policy (which discards the per-episode warm-start
        # state so eval never perturbs training), step_env's whole point is to
        # continue the real rollout across calls, so the final warm-start
        # state is persisted the same way the env state is.
        self.agent.policy_state = last_policy_state

        #reshape the jax laxed stuff (which includes a vmap over envs) into vectors of transitions
        transitions = self._to_transitions(num_steps, obs_sequence, action_sequence, next_obs_sequence,
                                           reward_sequence, dones)
        return transitions, last_obs, dones[-1]

    @jax.jit(static_argnames=["self", "policy"])
    def policy_rollout_scan(self, initial_obs, initial_state, policy_state, model_params, model_props, policy,
                            step_rngs):
        #step_body is defined here (not in rollout_policy) so it isn't a fresh
        #closure every call -- it's only rebuilt when this jitted method
        #actually retraces, not on every call to rollout_policy.
        def step_body(carry, step_rng):
            obs, env_state, policy_state = carry
            act_rng, step_rng = jax.random.split(step_rng)
            action, policy_state = policy(obs, act_rng, policy_state, model_params, model_props)
            next_obs, next_state, reward, done = self.env.pure_step(step_rng, env_state, action)
            return (next_obs, next_state, policy_state), (obs, action, next_obs, reward, done)

        carry = (initial_obs, initial_state, policy_state)
        return jax.lax.scan(step_body, carry, step_rngs)

    def rollout_policy(self, num_steps: int, policy: Callable, rng: jax.Array) -> Transition:
        """
        rollout a policy in the env. Env is reset before and after the rollout.
        :param num_steps: int, number of steps for rollout
        :param policy: Callable
        :param rng: jax.Array
        :return: transition: Transition from rollout
        """
        self.agent.prepare_agent_for_rollout() #reset policy_state (iCEM: zeroes) to restart planning from scratch
        rng, reset_rng = jax.random.split(rng, 2)
        reset_seed = jax.random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        if self._is_pure(self.env):
            step_rngs = jax.random.split(rng, num_steps)
            model_params, model_props = self.agent.get_model_inputs()
            (last_obs, last_state, last_policy_state), \
            (obs_sequence, action_sequence, next_obs_sequence,
            reward_sequence, dones) = self.policy_rollout_scan(
                obs, self.env._state, self.agent.policy_state, model_params, model_props, policy, step_rngs)
            sequences = (obs_sequence, action_sequence, next_obs_sequence, reward_sequence, dones)
        else:
            sequences, _, _, _ = self._host_rollout(obs, policy, num_steps, rng)

        transitions = self._to_transitions(num_steps, *sequences)
        reset_seed = jax.random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=num_steps).item()
        obs, _ = self.env.reset(seed=reset_seed)
        return transitions

    @jax.jit(static_argnames=["self", "env", "env_index"])
    def evaluation_scan(self, episode_rngs, policy_state, model_params, model_props, env, env_index):
        #step_body/evaluate_one_episode live here (not in better_eval_policy)
        #so they aren't fresh closures every call -- env/env_index are static,
        #so this whole method's jit cache persists across eval calls for the
        #same test env, and only retraces on an actual cache miss.
        def step_body(carry, rng):
            obs, env_state, policy_state, done = carry
            act_rng, env_rng = jax.random.split(rng)
            action, new_policy_state = self.agent.act_in_jax(obs, act_rng, policy_state, model_params, model_props,
                                                             eval=True, eval_idx=env_index)
            next_obs, next_state, reward, env_done = env.pure_step(env_rng, env_state, action)
            #convert the array given by logical_or to a scalar bool that can be read by lax.cond
            done = jax.lax.cond(jnp.logical_or(done, env_done).reshape(), lambda: 1, lambda: 0)
            return (next_obs, next_state, new_policy_state, done), (obs, action, next_obs, reward, done)

        #takes in an rng, returns the average reward from the environment
        #might break if the env is not compilable
        def evaluate_one_episode(rng, policy_state):
            reset_key, subkey = jax.random.split(rng)
            obs, initial_state = env.pure_reset(reset_key)
            step_rngs = jax.random.split(subkey, env.time_limit)
            #scan over steps: remove the ones after termination/truncation in calculating the average
            #policy_state is carried (iCEM warm start) within the episode, and dropped at its end
            (last_obs, last_state, policy_state, ever_terminated), (obs_sequence, action_sequence,
                                     next_obs_sequence, reward_sequence,
                                     done_sequence) = jax.lax.scan(step_body, (obs, initial_state, policy_state, 0), step_rngs)
            #reshape the reward sequence to be compatible with the done sequence
            reward_sequence = reward_sequence.reshape(-1)
            #done is sticky, so this is also the index of the step that ended the episode
            steps_before_termination = jnp.shape(done_sequence)[0] - jnp.count_nonzero(done_sequence)
            #rewards of the steps before the terminating step
            cumulative_reward = jnp.sum(jnp.where(done_sequence - 1, reward_sequence, jnp.zeros(jnp.shape(done_sequence))))
            #add the terminating step's reward, if the episode ended within the scan
            #(the index is clamped: under vmap, cond becomes a select that evaluates both branches)
            terminating_index = jnp.minimum(steps_before_termination, jnp.shape(done_sequence)[0] - 1)
            cumulative_reward += jax.lax.cond(jnp.count_nonzero(done_sequence) > 0,
                                              lambda: reward_sequence[terminating_index],
                                              lambda: jnp.zeros((), dtype=reward_sequence.dtype))
            return cumulative_reward #only check cumulative reward == undiscounted return

        evaluate_one_env = jax.vmap(evaluate_one_episode, in_axes=[0, None])
        return evaluate_one_env(episode_rngs, policy_state)

    def better_eval_policy(self, rng: jax.Array) -> dict:
        # test envs without pure_step can't be scanned: evaluate them all in eval_policy's host loop
        if not all(self._is_pure(env) for env in self.test_env):
            return self.eval_policy(rng=rng)

        self.agent.prepare_agent_for_rollout()
        model_params, model_props = self.agent.get_model_inputs()
        reward_log = {}

        for env_index, env in enumerate(self.test_env):
            episode_rngs = jax.random.split(rng, self.eval_episodes)
            average_rewards = self.evaluation_scan(episode_rngs, self.agent.policy_state, model_params, model_props,
                                                   env, env_index)
            final_reward_average = jnp.mean(average_rewards)
            reward_log[f"reward_task_{env_index}"] = final_reward_average

        return reward_log
        

    def eval_policy(self, step: int = 0, rng: Optional[jax.Array] = None) -> dict:
        """
        Evaluates policy in the environment till it is done.
        Only used when the environment is not pure

        :param step: Current step for evaluation
        :param rng: key
        :return reward_log: results log from evaluation
        """
        reward_log = {

        }
        rng = self.rng if rng is None else rng
        self.agent.prepare_agent_for_rollout()
        model_params, model_props = self.agent.get_model_inputs()
        #print(self.test_env)
        for i, env in enumerate(self.test_env):
            pbar = tqdm()
            ret = 0.0
            for e in range(self.eval_episodes):
                obs, _ = env.reset(seed=e)
                policy_state = self.agent.policy_state
                terminated = False
                truncated = False
                steps = 0
                while not (terminated or truncated):
                    rng, act_rng = jax.random.split(rng)
                    action, policy_state = self.agent.act_in_jax(obs, act_rng, policy_state,
                                                                 model_params, model_props, eval=True, eval_idx=i)
                    next_obs, reward, terminated, truncated, info = env.step(np.asarray(action))
                    ret += float(np.asarray(reward).sum())
                    obs = next_obs
                    steps += 1
                    pbar.update(1)
                    # print(steps)
            ret /= self.eval_episodes #average return over number of episodes
            reward_log['reward_task_' + str(i)] = ret
            pbar.close()
        return reward_log

    @property
    def num_reward_models(self):
        return 1
