import jax
import math

from opax.models.dynamics_model import DynamicsModel
from opax.optimizers.icem_trajectory_optimizer import ICemTO
from opax.optimizers.sac_based_optimizer import SACOptimizer
import gymnasium as gym
from opax.utils.replay_buffer import Transition, BufferState
from opax.utils.replay_buffer import JaxReplayBuffer as ReplayBuffer
from opax.utils.utils import get_idx
from opax.agents.dummy_agent import DummyAgent
import jax.numpy as jnp
from typing import Union, Optional, Dict, Any
from opax.models.hucrl_model import HUCRLModel
from opax.models.gp_dynamics_model import GPDynamicsModel
from jaxtyping import PyTree

Model_list = list[DynamicsModel]


class ModelBasedAgent(DummyAgent):
    """
    action_space: gym.spaces.box
    observation_space: gym.spaces.box
    dynamics_model: Union[DynamicsModel, Model_list], if multiple rewards a list of dynamics model is passed
    policy_optimizer_name: str, name for the policy optimizer used for control
    horizon: int, planning horizon for optimizer
    n_particles: int, number of particles used in planning
    reset_model: bool, if model should reset after every update
    reset_model_opt_state: bool, if model optimizer state should reset after every update
    calibrate_model: bool, if model calibration should be done
    init_function: bool, if init function for the model should be called
    optimizer_kwargs: Optional[Dict[str, Any]], kwargs for the model based policy optimizer
    reset_optimizer_params_for: int, number of initial training steps for which the policy optimizer is reset
    log_full_training: bool, if model training should be logged
    log_agent_training: bool, if agent training should be logged
    """
    def __init__(
            self,
            action_space: gym.spaces.box,
            observation_space: gym.spaces.box,
            dynamics_model: Union[DynamicsModel, Model_list],
            policy_optimizer_name: str = "CemTO",
            horizon: int = 100,
            n_particles: int = 10,
            reset_model: bool = False,
            reset_model_opt_state: bool = True,
            calibrate_model: bool = True,
            init_function: bool = True,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            reset_optimizer_params_for: int = 5,
            log_full_training: bool = False,
            log_agent_training: bool = True,
            *args,
            **kwargs,
    ):
        super(ModelBasedAgent, self).__init__(*args, **kwargs)
        self.action_space = action_space
        self.observation_space = observation_space
        if isinstance(dynamics_model, DynamicsModel):
            self.dynamics_model_list = [dynamics_model]
            self.num_dynamics_models = 1
        else:
            self.dynamics_model_list = dynamics_model
            self.num_dynamics_models = len(dynamics_model)
        assert policy_optimizer_name in ["CemTO", "TraJaxTO", "SacOpt", "iCemTO"], "Optimizer must be CEM, TraJax, " \
                                                                                   "SAC " \
                                                                                   "or iCem"

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        action_dim = self.action_space.shape
        if isinstance(self.dynamics_model, HUCRLModel) or isinstance(self.dynamics_model, GPDynamicsModel):
            action_dim = (self.action_space.shape[0] + self.observation_space.shape[0], )
        if policy_optimizer_name == 'iCemTO':
            self.policy_optimizer = ICemTO(
                    dynamics_model_list=self.dynamics_model_list,
                    horizon=horizon,
                    n_particles=n_particles,
                    action_dim=action_dim,
                    **optimizer_kwargs,
                )
        elif policy_optimizer_name == "SacOpt":
            self.policy_optimizer = SACOptimizer(
                dynamics_model_list=self.dynamics_model_list,
                horizon=horizon,
                action_dim=action_dim,
                **optimizer_kwargs,
            )
        else:
            raise NotImplementedError(f"{policy_optimizer_name} is no longer available, use iCemTO or SacOpt")
        self.n_particles = n_particles
        self.reset_model = reset_model
        self.reset_model_opt_state = reset_model_opt_state
        self.calibrate_model = calibrate_model
        self.reset_optimizer_params_for = reset_optimizer_params_for
        self.update_steps = 0
        self.buffer_state: Optional[BufferState] = None
        # Agent-owned copy of the policy optimizer's per-rollout state (iCEM: the
        # BestSequences warm start, SAC: the SacOptimizerState holding the trained
        # policies). Owned here (not by self.policy_optimizer) so act_in_jax can
        # thread it through as an explicit input/output instead of reading or
        # mutating optimizer attributes, which jit would freeze at first trace.
        self.policy_state = self.initial_policy_state()
        self.log_agent_training = log_agent_training
        self.log_full_training = log_full_training
        if init_function:
            self._init_fn()

    def _init_fn(self):
        """Creates model training function"""
        def step(carry, ins):
            rng = carry[0]
            model_params = carry[2]
            model_opt_state = carry[3]
            tran = ins[0]
            val_tran = ins[1]
            train_rng, rng = jax.random.split(rng, 2)

            (
                new_model_params,
                new_model_opt_state,
                alpha,
                summary,
            ) = \
                self.dynamics_model._train_step(
                    tran=tran,
                    model_params=model_params,
                    model_opt_state=model_opt_state,
                    val=val_tran,
                    rng=train_rng,
                )
            carry = [
                rng,
                alpha,
                new_model_params,
                new_model_opt_state,
            ]
            outs = [summary]
            return carry, outs

        self.step = jax.jit(step)
        
    @jax.jit(static_argnames=["self", "eval", "eval_idx"])
    def act_in_jax(self, obs: jax.Array, rng: jax.Array, policy_state, model_params, model_props,
                   eval: bool = False, eval_idx: int = 0):
        """Get an action from the policy optimizer.

        policy_state, model_params and model_props are explicit inputs because self is
        static here: anything read off self is frozen into the compiled function at the
        first trace, so updated model parameters or optimizer state would never be seen.
        The new policy_state is returned explicitly for the caller to carry forward.
        """
        # forward pass through policy if policy optimizer is SAC, its state is not changed by acting
        if isinstance(self.policy_optimizer, SACOptimizer):
            if eval:
                action = self.policy_optimizer.get_action_for_eval(obs=obs, rng=rng, agent_idx=eval_idx,
                                                                   policy_state=policy_state)
            else:
                action = self.policy_optimizer.get_action_for_exploration(obs=obs, rng=rng,
                                                                          policy_state=policy_state)
            action = action[..., :self.action_space.shape[0]]
            return action, policy_state

        # otherwise optimize for an action sequence via MPC and take its first action
        if eval:
            obs = obs.reshape(-1, obs.shape[-1])
            optimize_fn = self.policy_optimizer.optimize_for_eval_fns[eval_idx]
        else:
            optimize_fn = self.policy_optimizer.optimize_for_exploration

        n_envs = obs.shape[0]
        rollout_rng, optimizer_rng = jax.random.split(rng, 2)
        rollout_rng = jax.random.split(rollout_rng, n_envs)
        optimizer_rng = jax.random.split(optimizer_rng, n_envs)

        action_sequence, best_reward, new_policy_state = optimize_fn(
            dynamics_params=model_params,
            obs=obs,
            best_sequences=policy_state,
            key=rollout_rng,
            optimizer_key=optimizer_rng,
            model_props=model_props,
        )
        action = action_sequence[:, 0, ...]
        if eval and action.shape[0] == 1:
            action = action.squeeze(0)
        action = action[..., :self.action_space.shape[0]]
        return action, new_policy_state


    def train_step_gp(self, rng: jax.Array, buffer: ReplayBuffer, validate: bool=True,
                      log_results: bool=True) -> int:
        transitions = buffer.get_full_normalized_data()
        train_rng, rng = jax.random.split(rng, 2)
        model_params = self.dynamics_model.init_model_params
        model_opt_state = self.dynamics_model.init_model_opt_state
        alpha = jnp.ones(self.observation_space.shape)
        if self.num_epochs > 0:
            total_train_steps = math.ceil(buffer.size * self.num_epochs)
        else:
            total_train_steps = self.train_steps
        (
            model_params,
            model_opt_state,
            alpha,
            summary,
        ) = \
            self.dynamics_model._train_step(
                tran=transitions,
                model_params=model_params,
                model_opt_state=model_opt_state,
                rng=train_rng,
                num_steps=total_train_steps,
            )
        self.update_models(model_params=model_params, model_opt_state=model_opt_state, alpha=alpha)
        return total_train_steps
        
    @jax.jit(static_argnames=["self"])
    def train_step_scan(self, rng: jax.Array, alpha, model_params, model_opt_state, transitions, val_transitions):
        carry = [rng, alpha, model_params, model_opt_state]
        ins = transitions, val_transitions
    #pull this out to avoid recompilation of every gradient step
        carry, outs = jax.lax.scan(self.step, carry, ins)#, length=self.train_steps)
        return carry, outs
        

    def train_step(self,
                   rng: jax.Array,
                   buffer: ReplayBuffer,
                   buffer_state: BufferState,
                   validate: bool = True,
                   log_results: bool = True,
                   logger=None,
                   ) -> int:
        """
        Training of model based agent
        :param rng: jax.Array, random key
        :param buffer: ReplayBuffer, buffer of collected transition
        :param buffer_state: BufferState, current state (data) of the buffer to sample from
        :param validate: bool, boolean for validating model
        :param log_results: bool, boolean to indicate logging of training results.
        :param logger: CSVLogger, logger to record training results to, required if log_results is True
        :return
        """
        self.buffer_state = buffer_state

        # perform train steps on the NN model.
        max_train_steps_per_iter = 8000
        if self.num_epochs > 0:
            total_train_steps = math.ceil(self.buffer_state.size * self.num_epochs / self.batch_size)
        else:
            total_train_steps = self.train_steps
        self.update_steps += 1 #update, needs to be fixed to be compilable
        total_train_steps = min(total_train_steps, self.max_train_steps)
        train_loops = math.ceil(total_train_steps / max_train_steps_per_iter)
        train_steps = min(max_train_steps_per_iter, total_train_steps)
        train_loops = max(train_loops, 1)
        if self.reset_model:
            model_params = self.dynamics_model.init_model_params
            model_opt_state = self.dynamics_model.init_model_opt_state
        else:
            model_params = self.dynamics_model.model_params
            if self.reset_model_opt_state:
                model_opt_state = self.dynamics_model.init_model_opt_state
            else:
                model_opt_state = self.dynamics_model.model_opt_state
        alpha = jnp.ones(self.observation_space.shape)

        for i in range(train_loops):
            train_rng, rng = jax.random.split(rng, 2)
            transitions = buffer.sample(train_rng, self.buffer_state, batch_size=int(self.batch_size * train_steps))
            transitions = transitions.reshape(train_steps, self.batch_size)
            val_transitions = None
            if validate:
                train_rng, val_rng = jax.random.split(train_rng, 2)
                val_transitions = buffer.sample(val_rng, self.buffer_state, batch_size=int(self.batch_size * train_steps))
                val_transitions = val_transitions.reshape(train_steps, self.batch_size)
            carry = [
                train_rng,
                alpha,
                model_params,
                model_opt_state,
            ]
            ins = [transitions, val_transitions]
            carry, outs = self.train_step_scan(train_rng, alpha, model_params, model_opt_state, transitions, val_transitions)
            model_params = carry[2]
            model_opt_state = carry[3]
            alpha = carry[1]
            summary = outs[0]
            if log_results and logger is not None:
                if self.log_full_training:
                    for i in range(total_train_steps):
                        logger.log(get_idx(summary, i).dict())
                else:
                    for i in range(0, total_train_steps, 10):
                        logger.log(get_idx(summary, i).dict())
        if self.calibrate_model:
            alpha = carry[1]
        self.update_models(model_params=model_params, model_opt_state=model_opt_state, alpha=alpha)
        # train policy if SAC optimizer is used.
        if isinstance(self.policy_optimizer, SACOptimizer):
            if self.buffer_state.size > self.policy_optimizer.transitions_per_update:
                train_rng = carry[0]
                policy_train_rng, train_rng = jax.random.split(train_rng, 2)
                policy_agent_train_summary, policy_train_rewards = self.policy_optimizer.train(
                    rng=policy_train_rng,
                    true_obs=self.buffer_state.tran.obs,
                    obs_size=self.buffer_state.size,
                    dynamics_params=model_params,
                    model_props=self.dynamics_model.model_props,
                    reset_agent=self.reset_agent_params,
                )
                if log_results and self.log_agent_training and logger is not None:
                    for i in range(len(self.policy_optimizer.agent_list)):
                        summary_dict = policy_agent_train_summary[i].dict()
                        logger.log({key + '_agent_' + str(i): value for key, value in summary_dict.items()})
                        for reward in policy_train_rewards[i]:
                            logger.log({'sac_reward_agent_' + str(i): reward.item()})
        return total_train_steps

    def get_model_inputs(self):
        """Current (model_params, model_props) to pass into act_in_jax."""
        return self.dynamics_model.model_params, self.dynamics_model.model_props

    def initial_policy_state(self):
        """Policy state at the start of a rollout: iCEM's all-zeros warm start, or SAC's trained policies."""
        if isinstance(self.policy_optimizer, SACOptimizer):
            return self.policy_optimizer.optimizer_state
        return self.policy_optimizer.best_sequences

    def set_transforms(self,
                       bias_obs: Union[jnp.ndarray, float] = 0.0,
                       bias_act: Union[jnp.ndarray, float] = 0.0,
                       bias_out: Union[jnp.ndarray, float] = 0.0,
                       scale_obs: Union[jnp.ndarray, float] = 1.0,
                       scale_act: Union[jnp.ndarray, float] = 1.0,
                       scale_out: Union[jnp.ndarray, float] = 1.0,
                       ):
        """Set normalization factors for the dynamics models."""
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].set_transforms(
                bias_obs=bias_obs,
                bias_act=bias_act,
                bias_out=bias_out,
                scale_obs=scale_obs,
                scale_act=scale_act,
                scale_out=scale_out,
            )

    def predict_next_state(self,
                           tran: Transition,
                           ):
        """Predict next state with learned model."""
        return self.dynamics_model.predict_raw(
            parameters=self.dynamics_model.model_params,
            tran=tran,
            model_props=self.dynamics_model.model_props,
        )

    def update_models(self, model_params: PyTree, model_opt_state: PyTree, alpha: Union[float, jax.Array] = 1.0):
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].update_model(model_params=model_params,
                                                     model_opt_state=model_opt_state,
                                                     alpha=alpha,
                                                     model_props=self.dynamics_model_list[i].model_props)
        

    @property
    def dynamics_model(self) -> DynamicsModel:
        return self.dynamics_model_list[0]

    @property
    def reset_agent_params(self) -> bool:
        return self.update_steps < self.reset_optimizer_params_for

    def prepare_agent_for_rollout(self):
        # DummyPolicyOptimizer.reset() is a no-op; the optimizer state now lives
        # on the agent (self.policy_state), so resetting it for a fresh rollout
        # means restoring iCEM's initial all-zeros template, or picking up SAC's
        # latest trained policies, rather than calling into the optimizer.
        self.policy_state = self.initial_policy_state()

    def update_posterior(self, buffer: ReplayBuffer):
        for i in range(len(self.dynamics_model_list)):
            self.dynamics_model_list[i].update_model_posterior(buffer=buffer)

    @property
    def is_gp(self) -> bool:
        return isinstance(self.dynamics_model, GPDynamicsModel)
