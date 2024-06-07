import numpy as np
import itertools
import yaml

from .BaseHook import BaseHook
from .utils import make_env

class PandaHook(BaseHook):
    def __init__(self) -> None:
        self.max_step_num = 50
        # self.causal_keys = {'friction','mass','hidden_h','hidden_c'}
        self.causal_keys = {'hidden_h','hidden_c'}

    def start_test(self, train_envs, test_envs=None):

        # train_envs = eval(train_envs)
        # train_envs = list(set(train_envs))
        # self.train_envs = [(float(e[0]),float(e[1])) for e in train_envs]
        # self.train_envs.sort()
        # self.test_fricitions = list(set([e[0] for e in train_envs]))
        # self.test_fricitions.sort()
        # self.test_masses = list(set([e[1] for e in train_envs]))
        # self.test_masses.sort()
        # self.test_envs = list(itertools.product(self.test_fricitions, self.test_masses))
        
        train_envs = eval(train_envs)
        self.train_envs = train_envs
        test_envs = eval(test_envs) if test_envs is not None else []
        self.test_envs = test_envs
        # self.test_envs = train_envs + test_envs
        
        self.reward_table = np.zeros((len(self.test_envs),1))
        self.test_infos = {} # save to test

        # self.success_rate_table = np.zeros((len(self.test_fricitions), len(self.test_masses)))
        # self.pick_and_place_rate_table = np.zeros((len(self.test_fricitions), len(self.test_masses)))
        # self.roll_rate_table = np.zeros((len(self.test_fricitions), len(self.test_masses)))
        # self.push_rate_table = np.zeros((len(self.test_fricitions), len(self.test_masses)))

        self.test_infos = {} # save to test

    def start_env(self, env_info):
        test_info_key = self.encoder_env_info(env_info)
        self.test_infos[test_info_key] = {
            'success_rate':-1,
            'pick_and_place_rate':-1,
            'roll_rate':-1,
            'push_rate':-1,
            'eps_states':[]
        }

        self.all_count = 0
        self.success_count = 0
        self.pick_and_place_count = 0
        self.roll_count = 0
        self.push_count = 0

    def end_eps(self,env_info, _eps_states):
        brief_eps_states = self.states_to_string(_eps_states)
        result = self.states_to_result(_eps_states)
        if _eps_states[-1] == 'success':
            self.success_count += 1

        if result == 'pickandplace':
            self.pick_and_place_count += 1
        elif result == 'roll':
            self.roll_count += 1
        else:
            self.push_count += 1
        self.all_count += 1
        self.test_infos[self.encoder_env_info(env_info)]['eps_states'].append(brief_eps_states)
    
    def end_env(self, env_info, logger):
        test_info_key = self.encoder_env_info(env_info)
        self.test_infos[test_info_key]['success_rate'] = self.success_count / self.all_count
        self.test_infos[test_info_key]['pick_and_place_rate'] = self.pick_and_place_count / (self.all_count + 0.01)
        self.test_infos[test_info_key]['roll_rate'] = self.roll_count / (self.all_count + 0.01)
        self.test_infos[test_info_key]['push_rate'] = self.push_count / (self.all_count + 0.01)

        logger.info('#' * 10 + "friction:%.1f mass:%.1f "%env_info + '#'*10)
        logger.info(f'success_rate {self.success_count / self.all_count}')
        logger.info(f'pick_and_place_rate {self.pick_and_place_count / (self.all_count + 0.01)}')
        logger.info(f'roll_rate {self.roll_count / (self.all_count + 0.01)}')
        logger.info(f'push_rate {self.push_count / (self.all_count + 0.01)}')
        logger.info('#' * 19)

        # self.success_rate_table[self.test_fricitions.index(env_info[0]), self.test_masses.index(env_info[1])] = self.success_count / self.all_count
        # self.pick_and_place_rate_table[self.test_fricitions.index(env_info[0]), self.test_masses.index(env_info[1])] = self.pick_and_place_count / (self.success_count + 0.001)
        # self.roll_rate_table[self.test_fricitions.index(env_info[0]), self.test_masses.index(env_info[1])] = self.roll_count / (self.success_count + 0.001)
        # self.push_rate_table[self.test_fricitions.index(env_info[0]), self.test_masses.index(env_info[1])] = self.push_count / (self.success_count + 0.001)
        

    def end_hook(self, manager, time_steps):
        # row_names = [f"f{f}" for f in self.test_fricitions]
        # col_names = [f"m{m}" for m in self.test_masses]

        # manager.plot_table(self.success_rate_table, row_names, col_names, f'success_rate_{time_steps}')
        # manager.plot_table(self.pick_and_place_rate_table, row_names, col_names, f'pick_and_place_rate_{time_steps}')
        # manager.plot_table(self.roll_rate_table, row_names, col_names, f'roll_rate_{time_steps}')
        # manager.plot_table(self.push_rate_table, row_names, col_names, f'push_rate_{time_steps}')

        with open(manager.test_path, 'w') as f:
            yaml.dump(self.test_infos, f)

    def get_state(self,envs,env_info=None):
        object = envs.envs[0].unwrapped.sim._bodies_idx['object']
        table = envs.envs[0].unwrapped.sim._bodies_idx['table']
        robot = envs.envs[0].unwrapped.sim._bodies_idx['panda']
        contact_points = envs.envs[0].unwrapped.sim.physics_client.getContactPoints(bodyA=table, bodyB=object, linkIndexA=-1,linkIndexB = -1)
        contact_points1 = envs.envs[0].unwrapped.sim.physics_client.getContactPoints(bodyA = robot,bodyB = object, linkIndexA = 9, linkIndexB = -1)
        contact_points2 = envs.envs[0].unwrapped.sim.physics_client.getContactPoints(bodyA = robot,bodyB = object, linkIndexA = 10, linkIndexB = -1)
        object_info = envs.envs[0].unwrapped.sim.physics_client.getBasePositionAndOrientation(bodyUniqueId=object)
        fingers_width = envs.envs[0].unwrapped.robot.get_fingers_width()

        at_high = object_info[0][2] > 0.021 * 1
        # clamp_finger = fingers_width > 0.03 and fingers_width < 0.0405
        zero_table_contact = len(contact_points) == 0
        contact_with_two_fingers = (len(contact_points1) > 0 and len(contact_points2)) > 0
        
        if at_high and zero_table_contact and contact_with_two_fingers:
            return 'pickandplace'
        elif at_high:
            return 'roll'
        elif object_info[0][2] > 0.019 and object_info[0][2]<0.21:
            return 'push'
        else:
            return 'down'
    
    def states_to_result(self,states):
        for state in states:
            if 'pickandplace' in state:
                return 'pickandplace'
        for state in states:
            if 'roll' in state:
                return 'roll'
        return 'push'
    
    def states_to_string(self,states):
        tran_dict = {
            'roll':'r',
            'pickandplace':'P',
            'push':'p',
            'down':'d',
            'success':'s',
            'fail':'f',
        }
        _states = [tran_dict[s.split('_')[0]] for s in states]
        return ''.join(_states)

    def make_env(self, manager, env_info):
        return make_env('PandaPush-v3', 
                lateral_friction=env_info[0],
                mass=env_info[1], reward_type='dense',
                causal_dim=manager.model_parameters['causal_dim'], 
                causal_hidden_dim=manager.model_parameters['causal_hidden_dim']
        )
    
    def encoder_env_info(self,env_info):
        return f'friction:{env_info[0]},mass:{env_info[1]}'
