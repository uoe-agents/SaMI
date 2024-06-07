from abc import ABC, abstractmethod

class BaseHook(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def start_test(self, test_envs_) -> None:
        pass
    
    @abstractmethod
    def start_env(self, env_info) -> None:
        pass
    
    @abstractmethod
    def end_eps(self,env_info, _eps_states) -> None:
        pass
    
    @abstractmethod
    def end_env(self,env_info, logger) -> None:
        pass
    
    @abstractmethod
    def end_hook(self,manager, time_steps) -> None:
        pass
    
    @abstractmethod
    def get_state(self,envs,env_info=None) -> str:
        pass

    @abstractmethod
    def make_env(self,manager, env_info):
        pass

    @abstractmethod
    def encoder_env_info(self, env_info)->str:
        pass

