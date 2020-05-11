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

def distance_reward(distance):
    assert distance >= 0
    if distance < 10:
        return 1
    elif distance < 50:
        return -0.1*distance + 2
    else:
        return -3

class HighwayExitEnvNDD_DBV(HighwayExitEnvNDD):
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
        super(HighwayExitEnvNDD_DBV, self).__init__(config)

    def reset(self, given_ini=None):
        return super(HighwayExitEnvNDD_DBV, self).reset(given_ini=given_ini)

    def determine_controlled_bv(self):
        for vehicle in self.road.vehicles[1:]:
            vehicle.controlled = False
        self.controlled_bvs = self.road.get_BV(self.vehicle, self.controlled_bv_num, self.delete_BV_position)
        if len(self.controlled_bvs) != self.controlled_bv_num:
            # print("Vehicle less than preset controlled bv!")
            self.controlled_bv_num = len(self.controlled_bvs)
        assert len(self.controlled_bvs) == self.controlled_bv_num
        for vehicle in self.controlled_bvs:
            vehicle.controlled = True          

    def step(self, action):
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        assert len(bv_action) == self.controlled_bv_num
        weight = self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, exit_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag":cav_crash_flag, "bv_crash_flag":bv_crash_flag, "exit_flag":exit_flag, "bv_crash_index":bv_crash_index, "cav_action":cav_action, "bv_crash_index":bv_crash_index}
        # choose new controlled bv
        # self.determine_controlled_bv()
        # get reward and observation
        bv_reward, infos = self._reward_bv_multi(info, done)
        cav_reward = self._reward_cav(infos, done)
        observation = self.observe_cav_bv()
        reward = Reward(cav_reward=cav_reward, bv_reward=bv_reward)
        for bv in self.road.vehicles[1:]:
            bv.actual_action = False
        return observation, reward, done, infos, weight

    
    def _reward_bv_multi(self, info, done):
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
        for i in range(self.controlled_bv_num):
            car_name = "car"+str(i)
            infos[car_name] = {"Crash_respon":False}
        infos["scene_type"] = None
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
                            infos["scene_type"] = "BinC"
                            # print("BinC")
                            reward_team = 0
                            if bv.controlled:
                                control_bv_index = self.controlled_bvs.index(bv)
                                reward_bv_multi[control_bv_index] = 0
                                infos["car"+str(control_bv_index)]["Crash_respon"] = True
                        else:
                            if not (infos["scene_type"] and infos["scene_type"] == "BinC"):
                                infos["scene_type"] = "CinB"
                                # print("CinB")
                                reward_team = 0
                                if bv.controlled:
                                    control_bv_index = self.controlled_bvs.index(bv)
                                    reward_bv_multi[control_bv_index] = 0

            for bv_index in bv_crash_index:
                # We want to get every crash_bv with a distinct reward
                crash_bv = self.road.vehicles[bv_index]
                for vehicle in [self.road.vehicles[i] for i in bv_crash_index]:
                    if crash_bv.check_collision_flag(vehicle):
                        # We find the collision two vehicles: vehicle and crash_bv
                        infos["scene_type"] = "BinB"
                        #print("BinB")
                        reward_team = 0
                        # if crash_bv is not controlled, then we do not care
                        if crash_bv.controlled:
                            control_bv_index = self.controlled_bvs.index(crash_bv) # this index is in term of controlled bvs -- which is the obs and reward index
                            crash_bv_duty = self.check_crash_duty(crash_bv, vehicle)
                            if crash_bv_duty:
                                reward_bv_multi[control_bv_index] = 0
                                infos["car"+str(control_bv_index)]["Crash_respon"] = True
        elif exit_flag:
            reward_team = 0
            reward_bv_multi = [0] * self.controlled_bv_num
            infos["scene_type"] = "Cexit"
            #print("Cexit")
        elif done:
            assert (not bv_crash_flag) and (not cav_crash_flag)
            reward_bv_multi = [1] * self.controlled_bv_num
            infos["scene_type"] = "Cfail"
            #print("Cfail")
        else:
            reward_team = 0
            reward_bv_multi = [0] * self.controlled_bv_num
        # Delete all immediate reward
        
        reward_bv_multi_all = reward_bv_multi
        #reward_bv_multi_all = list(map(lambda x:x/100,reward_bv_multi_all))
        return reward_bv_multi_all, infos

