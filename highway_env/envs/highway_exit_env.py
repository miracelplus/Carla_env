from __future__ import division, print_function, absolute_import
import numpy as np
from highway_env.envs import HighwayExitEnv
import random
#from ENV_agent.agent import *
# import CAV_agent.agent as agent
from functools import reduce
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import global_val
import scipy.io
import bisect
from gym.envs.registration import register


class HighwayExitEnvNDD(HighwayExitEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "initial_spacing": 2,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "lanes_count": 3,
        "vehicles_count": global_val.random_initialization_vehicles_count,  # 15
        "minimum_distance": 15,  # 25
        "duration": 30
    }

    def __init__(self, mode="NDD",user="ENV"):
        super(HighwayExitEnvNDD, self).__init__()          
        self.lane_max = 2
        self.lane_min = 0
        self.min_position = 0
        self.max_position = 500
        self.min_velocity = 20
        self.max_velocity = 40
        self.mode = mode
        self.loss_list = []
        self.generate_vehicle_mode = "Random"
        self.user = "ENV"
        # Following : NDD initialize
        self.presum_list_forward = []  # CDF for car following forward
        self.presum_list_backward = [1]  # CDF for car following backward
        self.r_to_idx_dic, self.rr_to_idx_dic, self.v_to_idx_dic, self.v_back_to_idx_dic, self.acc_to_idx_dic = {}, {}, {}, {}, {}  # key: real value, value: idx
        self.speed_CDF = global_val.speed_CDF  # for both CF and FF
        config = self.DEFAULT_CONFIG.copy()
        self.reset()

    def _is_terminal(self):
        """
            The episode is over when:
            1. a collision occurs to CAV
            2. a collision occurs to BV
            2. when the access ramp has been passed.
        """
        pass

    def step(self, action, CAV_action=1):
        """
            Perform an action and step the environment dynamics.
            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        pass
    
    def _reward_bv(self, info, done):
        pass

    def _reward_cav(self, info, terminal):
        """
            When crashed receive -1, when exit highway receive 1, else 0
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        pass

    def _simulate(self, action=None, action_bv_original=None):
        """
            Perform several steps of simulation with constant action
        """
        pass
    
register(
    id='highway-exit-ndd-v0',
    entry_point='highway_env.envs:HighwayExitEnvNDD',
)
