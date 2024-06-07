from gymnasium import spaces
import numpy as np
from collections import OrderedDict

from .serializable import Serializable
from .make_env import make_env


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space