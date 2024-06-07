from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from copy import deepcopy

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import BaseModel
from line_profiler import profile
# 
class Encoder(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])

        self.lstm = nn.LSTM(self.observation_dim, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim,self.action_dim)

        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)

    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        x = th.cat(([x[_x] for _x in x]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        H,(h,c) = self.lstm(x, (h,c))
        logits = self.fc(th.relu(H)[np.arange(batch_size),0,:])
        return logits, (h.squeeze(0),c.squeeze(0))
    
    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        x = th.cat(([obs[_x] for _x in obs if _x not in {'hidden_c', 'hidden_h'}]),dim = -1)
        if len(x.shape) == 2:
            B,L = x.shape[0], 1
        elif len(x.shape) == 3:
            B,L = x.shape[:2]

        x = x.reshape(B*L,1, -1)
        h = obs['hidden_h'].reshape(1, B*L, -1)
        c = obs['hidden_c'].reshape(1, B*L, -1)
        H,(_,_) = self.lstm(x, (h,c))
        logits = self.fc(th.relu(H)).squeeze(1)
        return logits

class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        # my params 
        causal_keys: set = {'friction','mass'},
        causal_hidden_dim :int = 128,
        causal_out_dim: int = 6,
        causal_keys_dim: int =2,
    ):
        self.causal_keys = causal_keys
        self.causal_hidden_dim = causal_hidden_dim
        self.causal_out_dim = causal_out_dim
        self.causal_keys_dim = causal_keys_dim
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)

        # encoder
        trajectory_space = deepcopy(self.observation_space)
        del trajectory_space.spaces['causal']
        trajectory_space = deepcopy(trajectory_space)
        trajectory_space['action'] = spaces.Box(-10,10,(self.action_dim,),dtype=np.float32)

        causal_space = spaces.Box(-10,10,(self.causal_out_dim,),dtype=np.float32)
        self.encoder = Encoder(trajectory_space, causal_space, hidden_dim=self.causal_hidden_dim).to(self.device)
        self.encoder_target = Encoder(trajectory_space, causal_space, hidden_dim=self.causal_hidden_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_target.set_training_mode(False)
        self.encoder.optimizer = self.optimizer_class(
            self.encoder.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                causal_keys = self.causal_keys,
                causal_hidden_dim = self.causal_hidden_dim,
                causal_out_dim = self.causal_out_dim,
            )
        )
        return data
    
    def rnn_encoder_predict(self, observation):
        self.set_training_mode(False)

        assert len(observation['hidden_h'].shape) == 2
        assert len(observation['observation'].shape) == 2

        encoder_observation = {k:v for k,v in observation.items() if k not in self.causal_keys | {'causal','hidden_c','hidden_h'}}
        encoder_hidden_h = {'hidden_h': observation['hidden_h']}
        encoder_hidden_c = {'hidden_c': observation['hidden_c']}
        
        encoder_observation, _ = self.obs_to_tensor(encoder_observation)
        encoder_hidden_h, _ = self.obs_to_tensor(encoder_hidden_h)
        encoder_hidden_c, _ = self.obs_to_tensor(encoder_hidden_c)

        encoder_logits, (encoder_hidden_h, encoder_hidden_c) = \
            self.encoder.forward_one_step(encoder_observation, h=encoder_hidden_h, c=encoder_hidden_c)
        state = (encoder_logits.detach().cpu().numpy(),
                 encoder_hidden_h.detach().cpu().numpy(),
                 encoder_hidden_c.detach().cpu().numpy())
        return state
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        _observation = {}
        for key in observation:
            if key not in self.causal_keys | {'action', 'hidden_c', 'hidden_h'}:
                _observation[key] = observation[key]
        _observation, vectorized_env = self.obs_to_tensor(_observation)

        with th.no_grad():
            actions = self._predict(_observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # # Remove batch dimension if needed
        # if not vectorized_env:
        #     actions = actions.squeeze(axis=0)
        return actions, state

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        self.encoder.set_training_mode(mode)
        return super().set_training_mode(mode)