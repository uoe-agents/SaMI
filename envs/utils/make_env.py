import gymnasium

def get_push_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0,
                 gravity=-9.81, object_height=1.0, reward_type = 'dense',
                 control_type='ee', causal_dim = -1, causal_hidden_dim = -1 ,
                 train_time_steps=0):
    def _init():
        env = gymnasium.make(f'PandaPush-v3',
                             reward_type = reward_type,
                             control_type = control_type,
                             object_height = object_height,
                             causal_dim = causal_dim,
                             causal_hidden_dim = causal_hidden_dim)
        
        env.task.set_total_train_timesteps(train_time_steps)

        # env.unwrapped.task.goal_range_high[-1] = 0
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        # gravity
        env.unwrapped.sim.physics_client.setGravity(0, 0, gravity)
        env.set_obs_friction_mass(lateral_friction,mass)

        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env.reset()
        return env
    return _init

def get_pick_and_place_env(lateral_friction=1.0,spinning_friction=0.001,mass=1.0,
                           gravity=-9.81, object_height=1, reward_type = 'dense',
                           control_type='ee',causal_dim = -1, causal_hidden_dim = -1,
                           train_time_steps=0):
    def _init():
        env = gymnasium.make(f'PandaPickAndPlace-v3',
                             reward_type = reward_type,
                             control_type = control_type,
                             object_height = object_height,
                             causal_dim = causal_dim,
                             causal_hidden_dim = causal_hidden_dim)
        env.task.set_total_train_timesteps(train_time_steps)
        block_uid = env.unwrapped.sim._bodies_idx['object']
        env.unwrapped.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        # change table's friction
        env.unwrapped.sim.set_lateral_friction('table', -1, lateral_friction=lateral_friction)
        env.unwrapped.sim.set_spinning_friction('table', -1, spinning_friction=spinning_friction)
        # gravity
        env.unwrapped.sim.physics_client.setGravity(0, 0, gravity)
        env.set_obs_friction_mass(lateral_friction,mass)

        block_uid = env.unwrapped.sim._bodies_idx['object']
        print("Info of objects", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        block_uid = env.unwrapped.sim._bodies_idx['table']
        print("Info of Table", env.unwrapped.sim.physics_client.getDynamicsInfo(bodyUniqueId=block_uid, linkIndex=-1))
        env.reset()
        return env
    return _init

def get_ant_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'AntEnv', 
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_half_cheetah_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'HalfCheetahEnv',
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_slim_humanoid_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'SlimHumanoidEnv', 
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_humanoid_standup_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'HumanoidStandupEnv', 
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_hopper_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'HopperEnv', 
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_walker_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'WalkerEnv', 
                             mass_scale_set=mass_scale_set, 
                             damping_scale_set=damping_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_cripple_half_cheetah_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'CrippleHalfCheetahEnv', 
                             cripple_set=cripple_set, 
                             extreme_set=extreme_set,
                             mass_scale_set=mass_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_cripple_hopper_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'CrippleHopperEnv', 
                             cripple_set=cripple_set, 
                             extreme_set=extreme_set,
                             mass_scale_set=mass_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_cripple_walker_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'CrippleWalkerEnv', 
                             cripple_set=cripple_set, 
                             extreme_set=extreme_set,
                             mass_scale_set=mass_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init
    
def get_walker_hopper_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'WalkerHopperEnv', 
                             cripple_set=cripple_set, 
                             extreme_set=extreme_set,
                             mass_scale_set=mass_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_cripple_ant_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'CrippleAntEnv', 
                             cripple_set=cripple_set, 
                             extreme_set=extreme_set,
                             mass_scale_set=mass_scale_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_cartpole_env(force_set, length_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'Cartpoleenvs', 
                             force_set=force_set, 
                             length_set=length_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def get_pendulum_env(mass_set, length_set,causal_dim,causal_hidden_dim):
    def _init():
        env = gymnasium.make(f'Pendulumenvs', 
                             mass_set=mass_set, 
                             length_set=length_set,
                             causal_dim=causal_dim,
                             causal_hidden_dim=causal_hidden_dim)
        env.reset()
        return env
    return _init

def make_env(name,**kwargs):
    causal_dim=kwargs.get('causal_dim',-1)
    causal_hidden_dim=kwargs.get('causal_hidden_dim',-1)
    if name == "PandaPush-v3" or name == "PandaPickAndPlace-v3":
        lateral_friction=kwargs.get('lateral_friction',1.0)
        spinning_friction=kwargs.get('spinning_friction',0.001)
        mass=kwargs.get('mass',1.0)
        gravity=kwargs.get('gravity',-9.81)
        object_height=kwargs.get('object_height',1.0)
        reward_type=kwargs.get('reward_type','dense')
        control_type=kwargs.get('control_type','ee')
        train_time_steps=kwargs.get('train_time_steps',0)
        if name == "PandaPush-v3":
            return get_push_env(lateral_friction, spinning_friction, mass, gravity,object_height,reward_type,control_type,causal_dim,causal_hidden_dim,train_time_steps)
        elif name == "PandaPickAndPlace-v3":
            return get_pick_and_place_env(lateral_friction, spinning_friction, mass, gravity,object_height,reward_type,control_type,causal_dim,causal_hidden_dim,train_time_steps)
    elif name == "AntEnv" or name == 'HalfCheetahEnv' or name == 'SlimHumanoidEnv' or name == 'HumanoidStandupEnv' or name == 'HopperEnv'or name == 'WalkerEnv':
        mass_scale_set = kwargs.get('mass_scale_set', [0.85, 0.9, 0.95, 1.0])
        damping_scale_set = kwargs.get('damping_scale_set', [1.0])
        if name == 'AntEnv':
            return get_ant_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'HalfCheetahEnv':
            return get_half_cheetah_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'SlimHumanoidEnv':
            return get_slim_humanoid_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'HumanoidStandupEnv':
            return get_humanoid_standup_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'HopperEnv':
            return get_hopper_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'WalkerEnv':
            return get_walker_env(mass_scale_set, damping_scale_set,causal_dim,causal_hidden_dim)
    elif name == 'CrippleHalfCheetahEnv' or name == 'CrippleAntEnv' or name == 'CrippleHopperEnv' or name == 'CrippleWalkerEnv' or name == 'WalkerHopperEnv':
        cripple_set = kwargs.get('cripple_set', [0, 1, 2, 3])
        extreme_set = kwargs.get('extreme_set', [0])
        mass_scale_set = kwargs.get('mass_scale_set', [1.0])
        if name == 'CrippleHalfCheetahEnv':
            return get_cripple_half_cheetah_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'CrippleAntEnv':
            return get_cripple_ant_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'CrippleHopperEnv':
            return get_cripple_hopper_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'CrippleWalkerEnv':
            return get_cripple_walker_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim)
        elif name == 'WalkerHopperEnv':
            return get_walker_hopper_env(cripple_set, extreme_set, mass_scale_set,causal_dim,causal_hidden_dim)
    elif name == 'Cartpoleenvs':
        force_set = kwargs.get('force_set', [1.0])
        length_set = kwargs.get('length_set', [1.0])
        return get_cartpole_env(force_set, length_set,causal_dim,causal_hidden_dim)
    elif name == 'Pendulumenvs':
        mass_set = kwargs.get('mass_set', [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25])
        length_set = kwargs.get('length_set', [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25])
        return get_pendulum_env(mass_set, length_set,causal_dim,causal_hidden_dim)
    else:
        raise Exception("Unkown Environment in make_env")

