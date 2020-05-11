from __future__ import division, print_function, absolute_import
import numpy as np
from highway_env.envs import HighwayExitEnvNDD
import random
from functools import reduce
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import global_val
import scipy.io
import bisect
import heapq
from collections import namedtuple

Action = namedtuple("Action", ["cav_action", "bv_action"])
Observation = namedtuple("Observation", ["cav_observation", "bv_observation"])
Reward = namedtuple("Reward", ["cav_reward", "bv_reward"])


class HighwayExitEnvNDD_CAV(HighwayExitEnvNDD):
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
        "duration": 30,
        "controlled_bv_num": 1
    }

    def __init__(self, config):
        self.team_spirit = 0.1
        self.dis_pena_thre = 10
        super(HighwayExitEnvNDD_CAV, self).__init__(config)

    def reset(self):
        return super(HighwayExitEnvNDD_CAV, self).reset()

    def get_cav_observation(self):
        return self.Vehicle_list_to_nparray_matrix([self.get_single_vehicle_observation(self.vehicle, CAV_flag = True)]).flatten()

    def step(self, action):
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        #assert len(bv_action) == self.controlled_bv_num
        self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, exit_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag":cav_crash_flag, "bv_crash_flag":bv_crash_flag, "exit_flag":exit_flag, "bv_crash_index":bv_crash_index, "cav_action":cav_action}
        # get reward and observation
        cav_reward = self._reward_cav(exit_flag, done)
        bv_reward = None
        observation = self.observe_cav_bv()
        reward = Reward(cav_reward=cav_reward, bv_reward=bv_reward)
        for bv in self.road.vehicles[1:]:
            bv.actual_action = False
        return observation, reward, done, info
