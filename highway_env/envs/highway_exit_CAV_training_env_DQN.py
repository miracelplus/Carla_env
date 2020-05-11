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
from gym.envs.registration import register

Action = namedtuple("Action", ["cav_action", "bv_action"])
Observation = namedtuple("Observation", ["cav_observation", "bv_observation"])
Reward = namedtuple("Reward", ["cav_reward", "bv_reward"])

class HighwayExitEnvNDD_CAV_train_DQN(HighwayExitEnvNDD):
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
        "controlled_bv_num": 0
    }

    def __init__(self, config):
        self.team_spirit = 0.1
        self.env_obs_num = 10
        super(HighwayExitEnvNDD_CAV_train_DQN, self).__init__(config)

    def reset(self, given_ini=None):
        return super(HighwayExitEnvNDD_CAV_train_DQN, self).reset(given_ini=given_ini)

    def determine_controlled_bv(self):
        self.controlled_bvs = []
        if self.controlled_bv_num > 0:
            for vehicle in self.road.vehicles[1:]:
                vehicle.controlled = False
            # If there is no eligible BVs in the observation range, then just control 1 or even 0 bvs 
            near_bvs = self.road.get_BV_EU(self.vehicle, len(self.road.vehicles[1:]), self.cav_observation_range)
            for vehicle in near_bvs:
                if vehicle.lane_index[2] >= self.vehicle.lane_index[2]:
                    vehicle.controlled = True
                    self.controlled_bvs.append(vehicle)
                    if len(self.controlled_bvs) == self.controlled_bv_num:
                        break        

    # def Vehicle_list_to_nparray_matrix(self, Vehicle_list):
    #     array_list = []
    #     bv_obs = Vehicle_list
    #     input_matrix = np.zeros([len(bv_obs), 3])
    #     for i in range(len(bv_obs)):
    #         input_matrix[i, 0] = bv_obs[i].position[0]
    #         input_matrix[i, 1] = bv_obs[i].lane_index[2]
    #         input_matrix[i, 2] = bv_obs[i].velocity
    #     input_vector = input_matrix.flatten() 
    #     array_list.append(input_vector)
    #     return np.array(array_list)

    # # For fixed BVs
    # def get_single_bv_observation(self, bv):
    #     # this function get the local observation of the bv
    #     return self.get_single_vehicle_observation(bv, CAV_flag=False)
    # def get_bv_observation(self):
    #     if self.controlled_bv_num: 
    #         # get observation list for bv from nearest to most distance
    #         whole_bvs_observation = []
    #         for bv in self.controlled_bvs:
    #             bv_obs = self.get_single_bv_observation(bv)
    #             whole_bvs_observation += bv_obs

    #         for bv in self.controlled_bvs:
    #             bv_obs, action_indicator = self.get_single_bv_observation(bv)
    #             whole_bvs_observation.append([bv_obs, action_indicator])
    #         whole_bvs_observation, whole_action_indicator = self.Vehicle_list_to_nparray_matrix(whole_bvs_observation)


    #         # Add CAV
    #         whole_bvs_observation.insert(0, self.vehicle)
    #         whole_bvs_observation = self.Vehicle_list_to_nparray_matrix(whole_bvs_observation)
    #         return whole_bvs_observation

    def step(self, action):
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        if bv_action:
            assert len(bv_action) == self.controlled_bv_num
        weight = self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, exit_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag":cav_crash_flag, "bv_crash_flag":bv_crash_flag, "exit_flag":exit_flag, "bv_crash_index":bv_crash_index, "cav_action":cav_action}
        # get reward and observation and action indicator
        bv_reward, infos = self._reward_bv_DQN(info, done)
        if done:
            if not infos["scene_type"]:
                print("Get you!")
        cav_reward = self._reward_cav(infos, done)
        # choose new controlled bv
        self.determine_controlled_bv()
        observation_and_indicator = self.observe_cav_bv()
        reward = Reward(cav_reward=cav_reward, bv_reward=bv_reward)
        for bv in self.road.vehicles[1:]:
            bv.actual_action = False
        return observation_and_indicator, reward, done, infos, weight

    def _reward_bv_DQN(self, info, done):
        reward_bv_multi = [0]*self.controlled_bv_num
        reward_team = 0
        infos = {}
        # We have 6 cases to define an over scene
        # 1. BinB : bv into bv, no cav crash but bv crash
        # 2. BinC : bv into cav, cav & bv crash and bv responsibility
        # 3. CinB : cav into bv, cav & bv crash and cav responsibility
        # 4. Cexit : cav succesfully exit the environment
        # 5. Cfail : cav fails to exit the environment
        # 6. Default : None

        info_all = {}
        info_all["scene_type"] = None
        exit_flag = info["exit_flag"]
        bv_crash_flag = info["bv_crash_flag"]
        cav_crash_flag = info["cav_crash_flag"]
        bv_crash_index = info["bv_crash_index"]
        cav_action = info["cav_action"]

        # if there is collision
        if cav_crash_flag or bv_crash_flag:
            # If cav has crashes
            if cav_crash_flag:
                for bv in [self.road.vehicles[i] for i in bv_crash_index]:
                    # find the bv crashed with the CAV
                    if self.vehicle.check_collision_flag(bv):
                        bv_duty = self.check_crash_duty(bv, self.vehicle)
                        if bv_duty:
                            info_all["scene_type"] = "BinC"
                            reward_team = -1

                        else:
                            if not (info_all["scene_type"] and info_all["scene_type"] == "BinC"):
                                info_all["scene_type"] = "CinB"
                                reward_team = 0  # 30

            for bv_index in bv_crash_index:
                # We want to get every crash_bv with a distinct reward
                crash_bv = self.road.vehicles[bv_index]
                for vehicle in [self.road.vehicles[i] for i in bv_crash_index]:
                    if crash_bv.check_collision_flag(vehicle):
                        # We find the collision two vehicles: vehicle and crash_bv
                        info_all["scene_type"] = "BinB"
                        reward_team = -1
                        break
                if info_all["scene_type"] == "BinB":
                    break


        elif exit_flag:
            reward_team = 0 #-30
            info_all["scene_type"] = "Cexit"
        elif done:
            assert (not bv_crash_flag) and (not cav_crash_flag)
            reward_team = 0 #60
            info_all["scene_type"] = "Cfail"
        else:
            # distance_list = list(map(lambda x: np.sqrt((x.position[0] - self.vehicle.position[0])**2 + (x.position[1] - self.vehicle.position[1])**2), self.controlled_bvs))
            # whole_distance = sum(distance_list)
            # reward_team = 1 * (self.cav_observation_range * self.controlled_bv_num - whole_distance)/(self.cav_observation_range * self.controlled_bv_num)
            reward_team = 0
        # distance_list = list(map(lambda x: np.sqrt((x.position[0] - self.vehicle.position[0])**2 + (x.position[1] - self.vehicle.position[1])**2), self.controlled_bvs))
        # team_distance_reward = 0
        # for i in range(len(self.controlled_bvs)):
        #     bv = self.controlled_bvs[i]
        #     bv_distance = distance_list[i]
        #     team_distance_reward += distance_reward(bv_distance)/2
        # reward_team += team_distance_reward
        return reward_team, info_all