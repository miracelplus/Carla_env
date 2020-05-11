from __future__ import division, print_function
import numpy as np
import bisect
import scipy.io
import os
import scipy
import copy
import scipy.stats
import global_val
import sys
from tensorboardX import SummaryWriter
import time
import math
import highway_env
from highway_env.envs.highway_exit_BV_env import *
from highway_env.envs.highway_exit_Pure_NDD import *
import gym
import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle
import json

env_config = {"min_distance":0, "max_distance":1600, "min_velocity":20, "max_velocity":40, "min_lane":0, "max_lane":2, "cav_observation_range":100, "bv_observation_range":100, "controlled_bv_num":0, "cav_observation_num":10, "bv_observation_num":10, "generate_vehicle_mode":"NDD", "delete_BV_position":1600}
global_val.controlled_bv = env_config["controlled_bv_num"]
env = HighwayExitEnvNDD_Pure_NDD(env_config)


def inner_product(vector1, vector2):
    return vector1.x*vector2.x + vector1.y*vector2.y + vector1.z+vector2.z

# def vector_to_numpy_vector(vector):
#     return np.array([vector.x, vector.y, vector.z])

# def get_velocity_scalar(vehicle):
#     velocity = vehicle.get_velocity()
#     velocity_nparray = vector_to_numpy_vector(velocity)
#     return np.linalg.norm(velocity_nparray)

# def round_value_function(real_value, round_item):
#     if round_item == "speed":
#         value_list = conf.speed_list
#         value_dic = conf.v_to_idx_dic
#     elif round_item == "range":
#         value_list = conf.r_list
#         value_dic = conf.r_to_idx_dic
#     elif round_item == "range_rate":
#         value_list = conf.rr_list
#         value_dic = conf.rr_to_idx_dic

#     if round_item == "speed":
#         value_idx = bisect.bisect_left(value_list, real_value) 
#         value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
#         try:
#             assert value_idx <= (len(value_list)-1)
#             assert value_idx >= 0
#         except:
#             print("Fxxk!!!")
#         round_value = value_list[value_idx]
#         assert value_dic[round_value] == value_idx
#         return round_value, value_idx
#     else: 
#         value_idx = bisect.bisect_left(value_list, real_value) 
#         value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
#         try:
#             assert value_idx <= (len(value_list)-1)
#             assert value_idx >= 0
#         except:
#             print("Fxxk!!!")
#         round_value = value_list[value_idx]
#         assert value_dic[round_value] == value_idx
#         return round_value, value_idx

# def round_value_lane_change(real_value, value_list, round_item="speed"):
#     if round_item == "speed":
#         value_idx = bisect.bisect_left(value_list, real_value) 
#         value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
#         try:
#             assert value_idx <= (len(value_list)-1)
#             assert value_idx >= 0
#         except:
#             print("Fxxk!!!")
#         round_value = value_list[value_idx]
#         return round_value, value_idx
#     else: 
#         value_idx = bisect.bisect_left(value_list, real_value) 
#         value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
#         try:
#             assert value_idx <= (len(value_list)-1)
#             assert value_idx >= 0
#         except:
#             print("Fxxk!!!")
#         round_value = value_list[value_idx]
#         return round_value, value_idx


def get_carla_observation(ego_idx, vehicle_list, waypoint_list, candidate_idx_list):
    f1, r1, f0, r0, f2, r2 = None, None, None, None, None, None
    f1_range, r1_range, f0_range, r0_range, f2_range, r2_range = None, None, None, None, None, None
    range_tmp = None
    v_scalar = get_velocity_scalar(vehicle_list[ego_idx])
    min_f0_range_value, min_f1_range_value, min_f2_range_value = np.inf, np.inf, np.inf
    min_r0_range_value, min_r1_range_value, min_r2_range_value = -np.inf, -np.inf, -np.inf

    ego_lane_id = waypoint_list[ego_idx].lane_id
    ego_lane_change_type = str(waypoint_list[ego_idx].lane_change)
    # get left/right id
    left_lane_id = waypoint_list[ego_idx].left_lane().lane_id
    right_lane_id = waypoint_list[ego_idx].right_lane().lane_id
    ego_velocity_vector = vehicle_list[ego_idx].get_velocity()
    for idx in candidate_idx_list[1:]:
        if idx != ego_idx:
            if waypoint_list[idx].lane_id == ego_lane_id:
                new_f_value = inner_product((waypoint_list[idx].transform.location-waypoint_list[ego_idx].transform.location), ego_velocity_vector)
                if new_f_value >= 0:
                    if new_f_value < min_f1_range_value:
                        min_f1_range_value = new_f_value
                        f1 = idx
                        f1_range = min_f1_range_value/v_scalar
                else:
                    if new_f_value > min_r1_range_value:
                        min_r1_range_value = new_f_value
                        r1 = idx
                        r1_range = min_r1_range_value/v_scalar                        
            elif waypoint_list[idx].lane_id == left_lane_id:
                new_f_value = inner_product((waypoint_list[idx].transform.location-waypoint_list[ego_idx].transform.location), ego_velocity_vector)
                if new_f_value >= 0:
                    if new_f_value < min_f0_range_value:
                        min_f0_range_value = new_f_value
                        f0 = idx
                        f0_range = min_f0_range_value/v_scalar
                else:
                    if new_f_value > min_r0_range_value:
                        min_r0_range_value = new_f_value
                        r0 = idx
                        r0_range = min_r0_range_value/v_scalar      
            elif waypoint_list[idx].lane_id == right_lane_id:
                new_f_value = inner_product((waypoint_list[idx].transform.location-waypoint_list[ego_idx].transform.location), ego_velocity_vector)
                if new_f_value >= 0:
                    if new_f_value < min_f2_range_value:
                        min_f2_range_value = new_f_value
                        f2 = idx
                        f2_range = min_f2_range_value/v_scalar
                else:
                    if new_f_value > min_r2_range_value:
                        min_r2_range_value = new_f_value
                        r2 = idx
                        r2_range = min_r2_range_value/v_scalar  
    return [f1, r1, f0, r0, f2, r2], [f1_range, r1_range, f0_range, r0_range, f2_range, r2_range]


