import os

from .method_step import SAC as OURSACStep
from .method_step import DictReplayBuffer as OURDictReplayBufferStep
from .method_step import MultiInputPolicy as OURMultiInputPolicyStep

from .method_traj import SAC as OURSACTraj
from .method_traj import DictReplayBuffer as OURDictReplayBufferTraj
from .method_traj import MultiInputPolicy as OURMultiInputPolicyTraj


def get_model(manager,train_env=None, causal_keys = None, max_step_num=50):
    if train_env is not None:
        kwargs = dict(
            policy=None,
            env = train_env,
            buffer_size=manager.model_parameters['buffer_size'], 
            causal_keys=causal_keys, # do not input rl policy
            causal_hidden_dim=manager.model_parameters['causal_hidden_dim'],
            causal_out_dim=manager.model_parameters['causal_dim'],
            batch_size=manager.model_parameters['batch_size'],
            gamma=0.95,
            learning_rate=manager.model_parameters['learning_rate'],
            train_freq=manager.model_parameters['train_freq'],
            gradient_steps=manager.model_parameters['gradient_steps'],
            tau=0.05,
            replay_buffer_class=None,
            policy_kwargs=dict(
                net_arch=[512, 512, 512],
                n_critics=2,
                causal_hidden_dim=manager.model_parameters['causal_hidden_dim'],
                causal_out_dim=manager.model_parameters['causal_dim'],
            ),
            replay_buffer_kwargs=dict(
                causal_keys=causal_keys,
                buffer_type=manager.model_parameters['contrast_buffer_type'],
                max_eps_length=max_step_num,
                neg_radio=manager.model_parameters['neg_radio'],
            ),
            learning_starts = 2000 * train_env.num_envs,
            verbose=1,
            encoder_tau = manager.model_parameters['encoder_tau'],
            adversarial_loss_coef = manager.model_parameters['adversarial_loss_coef'],
            use_weighted_info_nce= manager.model_parameters['use_weighted_info_nce'],
            contrast_batch_size = manager.model_parameters['contrast_batch_size'],
            contrast_training_interval = manager.model_parameters['contrast_training_interval']
        )
    if manager.model_parameters['method'] == 'TESAC':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['contrast_batch_size'] = 0
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    if manager.model_parameters['method'] == 'CCM':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['replay_buffer_kwargs']['buffer_type'] = 'env_diff'
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    if manager.model_parameters['method'] == 'SaCCM':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['replay_buffer_kwargs']['buffer_type'] = 'half'
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    if manager.model_parameters['method'] == 'SaCCM2':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['replay_buffer_kwargs']['buffer_type'] = 'fine_grained'
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    if manager.model_parameters['method'] == 'SaSAC':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['replay_buffer_kwargs']['buffer_type'] = 'one'
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    if manager.model_parameters['method'] == 'SaCCMAll':
        if not os.path.exists(manager.save_model_path):
            kwargs['policy'] = OURMultiInputPolicyTraj
            kwargs['replay_buffer_class'] = OURDictReplayBufferTraj
            kwargs['replay_buffer_kwargs']['buffer_type'] = 'all'
            model = OURSACTraj(**kwargs)
        else:
            model = OURSACTraj.load(manager.save_model_path,env=train_env)
    return model



# def get_model(manager,train_env=None, causal_keys = None, max_step_num=50):
#     if train_env is not None:
#         kwargs = dict(
#             policy=None,
#             env = train_env,
#             buffer_size=manager.model_parameters['buffer_size'] * 1000, 
#             causal_keys=causal_keys, # do not input rl policy
#             causal_hidden_dim=manager.model_parameters['causal_hidden_dim'],
#             causal_out_dim=manager.model_parameters['causal_dim'],
#             batch_size=manager.model_parameters['batch_size'] * 100,
#             gamma=0.95,
#             learning_rate=manager.model_parameters['learning_rate'],
#             train_freq=manager.model_parameters['train_freq'],
#             gradient_steps=manager.model_parameters['gradient_steps'],
#             tau=0.05,
#             replay_buffer_class=None,
#             policy_kwargs=dict(
#                 net_arch=[512, 512, 512],
#                 n_critics=2,
#                 causal_hidden_dim=manager.model_parameters['causal_hidden_dim'],
#                 causal_out_dim=manager.model_parameters['causal_dim']
#             ),
#             replay_buffer_kwargs=dict(
#                 causal_keys=causal_keys,
#                 buffer_type=manager.model_parameters['contrast_buffer_type'],
#             ),
#             learning_starts = 1000 * train_env.num_envs,
#             verbose=1,
#             encoder_tau = manager.model_parameters['encoder_tau'],
#             adversarial_loss_coef = manager.model_parameters['adversarial_loss_coef'],
#             use_weighted_info_nce= manager.model_parameters['use_weighted_info_nce'],
#             contrast_batch_size = manager.model_parameters['contrast_batch_size'],
#         )
#     if manager.model_parameters['method'] == 'TESAC':
#         if not os.path.exists(manager.save_model_path):
#             kwargs['policy'] = OURMultiInputPolicyStep
#             kwargs['replay_buffer_class'] = OURDictReplayBufferStep
#             kwargs['contrast_batch_size'] = 0
#             model = OURSACStep(**kwargs)
#         else:
#             model = OURSACStep.load(manager.save_model_path,env=train_env)
#     if manager.model_parameters['method'] == 'CCM':
#         if not os.path.exists(manager.save_model_path):
#             kwargs['policy'] = OURMultiInputPolicyStep
#             kwargs['replay_buffer_class'] = OURDictReplayBufferStep
#             kwargs['replay_buffer_kwargs']['buffer_type'] = 'env_diff'
#             model = OURSACStep(**kwargs)
#         else:
#             model = OURSACStep.load(manager.save_model_path,env=train_env)
#     if manager.model_parameters['method'] == 'SaCCM':
#         if not os.path.exists(manager.save_model_path):
#             kwargs['policy'] = OURMultiInputPolicyStep
#             kwargs['replay_buffer_class'] = OURDictReplayBufferStep
#             kwargs['replay_buffer_kwargs']['buffer_type'] = 'half'
#             model = OURSACStep(**kwargs)
#         else:
#             model = OURSACStep.load(manager.save_model_path,env=train_env)
#     if manager.model_parameters['method'] == 'SaSAC':
#         if not os.path.exists(manager.save_model_path):
#             kwargs['policy'] = OURMultiInputPolicyStep
#             kwargs['replay_buffer_class'] = OURDictReplayBufferStep
#             kwargs['replay_buffer_kwargs']['buffer_type'] = 'one'
#             model = OURSACStep(**kwargs)
#         else:
#             model = OURSACStep.load(manager.save_model_path,env=train_env)
#     if manager.model_parameters['method'] == 'SaCCMAll':
#         if not os.path.exists(manager.save_model_path):
#             kwargs['policy'] = OURMultiInputPolicyStep
#             kwargs['replay_buffer_class'] = OURDictReplayBufferStep
#             kwargs['replay_buffer_kwargs']['buffer_type'] = 'all'
#             model = OURSACStep(**kwargs)
#         else:
#             model = OURSACStep.load(manager.save_model_path,env=train_env)
#     return model

__all__ = ['get_model']
