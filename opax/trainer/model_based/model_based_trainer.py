import jax.random
from opax.utils.replay_buffer import JaxReplayBuffer as ReplayBuffer
from opax.utils.replay_buffer import Transition, BufferState
from opax.trainer.dummy_trainer import DummyTrainer
from opax.agents.model_based.model_based_agent import ModelBasedAgent
import wandb
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class ModelBasedTrainer(DummyTrainer):
    def __init__(self,
                 agent: ModelBasedAgent,
                 agent_name: str = "ModelBasedAgent",
                 validation_buffer_size: int = 0,
                 validation_batch_size: int = 1024,
                 uniform_exploration: bool = False,
                 *args,
                 **kwargs,
                 ):
        """

        :param agent: ModelBasedAgent, RL Agent
        :param agent_name: str, name of agent
        :param validation_buffer_size: int, size of validation buffer
        :param validation_batch_size: int, batch size for validation
        :param uniform_exploration: bool, if random/uniform exploration is done
        :param args:
        :param kwargs:
        """

        super(ModelBasedTrainer, self).__init__(agent=agent, agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, ModelBasedAgent), "Only Model based agents are allowed"
        self.validation_buffer = None
        self.validation_batch_size = validation_batch_size
        self.collect_validation_data(validation_buffer_size)
        self.uniform_exploration = uniform_exploration

    def collect_validation_data(self, validation_buffer_size: int = 0):
        """
        Collect data for validation if validation_buffer_size > 0
        :param validation_buffer_size: int
        :return:
        """
        if validation_buffer_size > 0:
            self.validation_buffer = ReplayBuffer(
                obs_shape=self.buffer.obs_shape,
                action_shape=self.buffer.action_shape,
                normalize=False,
                action_normalize=False,
                learn_deltas=True
            )
            
            self.validation_buffer_state = self.validation_buffer.initialize_buffer_state()
            
            num_points = int(validation_buffer_size * self.num_envs)
            obs_shape = (num_points,) + self.env.observation_space.shape
            action_space = (num_points,) + self.env.action_space.shape
            obs_vec = np.zeros(obs_shape)
            action_vec = np.zeros(action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros(obs_shape)
            done_vec = np.zeros((num_points,), dtype=bool)
            for step in range(validation_buffer_size):
                obs = self.env.observation_space.sample()
                action = self.env.action_space.sample()
                obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
                action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action

            transitions = Transition(
                obs=obs_vec,
                action=action_vec,
                reward=reward_vec,
                next_obs=next_obs_vec,
                done=done_vec,
            )

            #policy = lambda x, y: np.concatenate(
            #    [self.env.action_space.sample().reshape(1, -1)
            #     for s in range(self.num_envs)], axis=0)
            #self.rng, val_rng = random.split(self.rng, 2)
            #transitions = self.rollout_policy(validation_buffer_size,
            #                                  policy,
            #                                  val_rng
            #                                  )
            self.validation_buffer_state = self.validation_buffer.add(transitions, self.validation_buffer_state)

    @staticmethod
    @jax.jit(static_argnames=["is_gp"])
    def get_validation_stats(mean_prediction, std_prediction, is_gp):
        if is_gp:
            eps_uncertainty = std_prediction
            al_uncertainty = jnp.ones_like(std_prediction) * 0.001
        else:
            eps_uncertainty = jnp.std(mean_prediction, axis=0)
            al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std_prediction), axis=0))
        frac = eps_uncertainty / (al_uncertainty + 1e-6)
        information_gain = jnp.sum(jnp.log(1 + jnp.square(frac)), axis=-1)
        max_info_gain = jnp.max(information_gain)
        mean_info_gain = jnp.mean(information_gain)
        eps_uncertainty = jnp.sum(eps_uncertainty, axis=-1)
        mean_eps_uncertainty = jnp.mean(eps_uncertainty)
        max_eps_uncertainty = jnp.max(eps_uncertainty)
        std_eps_uncertainty = jnp.std(eps_uncertainty)
        std_pred = jnp.mean(jnp.sum(al_uncertainty, axis=-1))        
        return eps_uncertainty, al_uncertainty, frac, information_gain, max_info_gain, \
               mean_info_gain, eps_uncertainty, mean_eps_uncertainty, max_eps_uncertainty, \
               std_eps_uncertainty, std_pred
            

    def validate_model(self, rng: jax.Array) -> dict:
        """
        Validates learned dynamics model
        :param rng: random key for validation
        :return model_log: dict, logs for model validation
        """
        model_log = {}
        if self.validation_buffer is not None:
            val_tran = self.validation_buffer.sample(
                rng=rng,
                state=self.validation_buffer_state,
                batch_size=self.validation_batch_size,
            )
            mean_pred, std_pred = self.agent.predict_next_state(val_tran)
            eps_uncertainty, al_uncertainty, frac, information_gain, max_info_gain, \
            mean_info_gain, eps_uncertainty, mean_eps_uncertainty, max_eps_uncertainty, \
            std_eps_uncertainty, std_pred = self.get_validation_stats(mean_pred, std_pred, self.agent.is_gp)

            # y_true = val_tran.next_obs
            # mse = jnp.mean(jnp.sum(jnp.square(y_true - mean_pred), axis=-1))
            model_log = {
                # 'validation_mse': mse.astype(float).item(),
                'validation_al_std': std_pred.astype(float).item(),
                'validation_eps_std_mean': mean_eps_uncertainty.astype(float).item(),
                'validation_eps_std_max': max_eps_uncertainty.astype(float).item(),
                'validation_eps_std_std': std_eps_uncertainty.astype(float).item(),
                'max_info_gain': max_info_gain.astype(float).item(),
                'mean_info_gain': mean_info_gain.astype(float).item(),
            }
        return model_log

    def train(self):
        """
        Train function for model based agent
        """
        if self.use_wandb:
            wandb.define_metric('env_steps')
            wandb.define_metric('learning_step')
        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)

        self.agent.set_transforms(
                bias_obs=self.buffer_state.state_normalizer_state.mean,
                bias_act=self.buffer_state.action_normalizer_state.mean,
                bias_out=self.buffer_state.next_state_normalizer_state.mean,
                scale_obs=self.buffer_state.state_normalizer_state.std,
                scale_act=self.buffer_state.action_normalizer_state.std,
                scale_out=self.buffer_state.next_state_normalizer_state.std,
        )
        curr_eval, eval_val_rng = jax.random.split(curr_eval, 2)
        model_log = self.validate_model(eval_val_rng)
        reward_log = self.better_eval_policy(rng=curr_eval)
        best_performance = reward_log['reward_task_0']
        reward_log['env_steps'] = 0
        reward_log['learning_step'] = 0
        reward_log['train_steps'] = 0
        reward_log['update_steps_per_iter'] = 0
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.define_metric("env_steps")
            wandb.define_metric("train_steps")
            reward_log.update(model_log)
            wandb.log(reward_log)
            #get the dynamics MAE before any learning
            #wandb.log({
            #    'env_steps': 0,
            #    **self.agent.dynamics_model.get_dynamics_mae(rng=self.rng),
            #})

        # returns policy used for random exploration
        #may need to be changed based on the env but I think the actions are always clipped to [-1,1] for normalization?
        # best_sequences is passed through unchanged: random exploration doesn't
        # touch the iCEM warm-start state, but every policy callable passed to
        # step_env/rollout_policy needs the same (obs, rng, best_sequences) ->
        # (action, best_sequences) shape so those methods can stay agnostic to
        # which kind of policy they were given.
        exploration_policy = lambda obs, rng, best_sequences: (
            jax.random.uniform(rng, self.action_shape, minval=-1.0, maxval=1.0), best_sequences
        )
        # collect data with the random policy
        policy = exploration_policy
        self.rng, explore_rng = random.split(self.rng, 2)
        if self.exploration_steps > 0:
            transitions = self.rollout_policy(self.exploration_steps, policy, explore_rng)
            self.buffer_state = self.buffer.add(transitions, self.buffer_state)
        rng_keys = random.split(self.rng, self.total_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.total_train_steps / (self.rollout_steps * self.num_envs))
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps * self.rollout_steps)).item()
        # reset env before training starts
        obs, _ = self.env.reset(seed=reset_seed)
        step = 0
        for step in tqdm(range(learning_steps)):
            print(f"Step: {step}")
            # collect rollouts with policy
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act_in_train if not self.uniform_exploration else exploration_policy
            actor_rng, val_rng = random.split(actor_rng, 2)
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer_state = self.buffer.add(transitions, self.buffer_state)
            self.agent.set_transforms(
                bias_obs=self.buffer_state.state_normalizer_state.mean,
                bias_act=self.buffer_state.action_normalizer_state.mean,
                bias_out=self.buffer_state.next_state_normalizer_state.mean,
                scale_obs=self.buffer_state.state_normalizer_state.std,
                scale_act=self.buffer_state.action_normalizer_state.std,
                scale_out=self.buffer_state.next_state_normalizer_state.std,
            )
            self.agent.update_posterior(self.buffer)
            reward_log = {}
            train_step_log = {}
            model_log = {}
            env_step_log = {
                'env_steps': step * self.rollout_steps * self.num_envs,
                'learning_step': step,
            }
            # update agent
            if step % self.train_freq == 0 and (self.buffer_state.size >= self.agent.batch_size or self.agent.is_gp):
                train_rng, agent_rng = random.split(train_rng, 2)
                total_train_steps = self.agent.train_step(
                    rng=agent_rng,
                    buffer=self.buffer,
                    buffer_state=self.buffer_state,
                    validate=self.validate,
                    log_results=self.use_wandb,
                )
                train_steps += total_train_steps
                train_step_log = {
                    'train_steps': train_steps,
                    'update_steps_per_iter': total_train_steps,
                }
                eval_rng, eval_val_rng = jax.random.split(eval_rng, 2)
                model_log = self.validate_model(eval_val_rng)
            # Evaluate episode
            if step % self.eval_freq == 0 and train_steps > 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                reward_log = self.better_eval_policy(rng=curr_eval)
                if reward_log['reward_task_0'] > best_performance:
                    best_performance = reward_log['reward_task_0']
                    self.save_agent(step)

            if self.use_wandb:
                train_log = env_step_log
                train_log.update(train_step_log)
                train_log.update(reward_log)
                train_log.update(model_log)
                #scalar_dict = {'scale_out': np.mean(self.buffer.next_state_normalizer.std).astype(float).item()}
                # scaler_dict = {
                #     'bias_obs': np.mean(self.buffer.state_normalizer.mean).astype(float).item(),
                #     'bias_act': np.mean(self.buffer.action_normalizer.mean).astype(float).item(),
                #     'bias_out': np.mean(self.buffer.next_state_normalizer.mean).astype(float).item(),
                #     'scale_obs': np.mean(self.buffer.state_normalizer.std).astype(float).item(),
                #     'scale_act': np.mean(self.buffer.action_normalizer.std).astype(float).item(),
                #     'scale_out': np.mean(self.buffer.next_state_normalizer.std).astype(float).item(),
                # }
                #train_log.update(scalar_dict)
                wandb.log(train_log)
                wandb.log({
                    'env_steps': env_step_log['env_steps'],
                    **self.agent.dynamics_model.get_dynamics_mae(rng=self.rng),
                })
            # step += 1
        self.save_agent(step, agent_name="final_agent")

    @property
    def num_reward_models(self) -> int:
        return self.agent.num_dynamics_models
