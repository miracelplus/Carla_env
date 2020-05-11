from __future__ import division, print_function, absolute_import
import numpy as np
from highway_env.envs import HighwayExitEnvNDD
import random
from functools import reduce
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, Pure_NDDVehicle, IDMVehicle_CAV
import global_val
import scipy.io
import bisect
import heapq
from collections import namedtuple
from gym.envs.registration import register
from highway_env import utils

Action = namedtuple("Action", ["cav_action", "bv_action"])
Observation = namedtuple("Observation", ["cav_observation", "bv_observation"])
Reward = namedtuple("Reward", ["cav_reward", "bv_reward"])

class HighwayExitEnvNDD_CAV_train_DQN_Safety(HighwayExitEnvNDD):
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
        super(HighwayExitEnvNDD_CAV_train_DQN_Safety, self).__init__(config)

    def reset(self, given_ini=None):
        return super(HighwayExitEnvNDD_CAV_train_DQN_Safety, self).reset(given_ini=given_ini)

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

    def step(self, action):
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        if bv_action:
            assert len(bv_action) == self.controlled_bv_num
        weight = self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, range_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag":cav_crash_flag, "bv_crash_flag":bv_crash_flag, "finish_flag":range_flag, "bv_crash_index":bv_crash_index, "cav_action":cav_action}
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

    def get_cav_observation(self):
        # !!! Here do not use safety check yet.
        action_indicator = self.vehicle.get_action_indicator(ndd_flag = False, safety_flag = True, CAV_flag = True)
        obs, cav_obs_vehs_list = self.observation.original_observe_acc_training(cav_obs_num=self.cav_observation_num, cav_observation_range=self.cav_observation_range)
        self.cav_obs_vehs_list = cav_obs_vehs_list
        return obs, action_indicator

    def _reward_cav(self, infos, terminal):
        """
            When crashed receive -1, when exit highway receive 1, else 0
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        CinB_reward = 0
        BinC_reward = 0
        High_velocity_reward = 1  # The reward received when driving at 40 m/s, linearly mapped to zero for 35 m/s
        # speed_reward = np.clip(utils.remap(self.vehicle.velocity, [30, 40], [0, High_velocity_reward]), 0, High_velocity_reward)
        speed_reward = np.clip(utils.remap(self.vehicle.velocity, [20, 40], [-High_velocity_reward, High_velocity_reward]), -High_velocity_reward, High_velocity_reward)        
        scene_type = infos["scene_type"]
        if not scene_type:
            return speed_reward
        else:
            if scene_type == "CinB":
                return CinB_reward
            elif scene_type == "BinC":
                return BinC_reward
            else:
                return 0

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
        finish_flag = info["finish_flag"]
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

        elif finish_flag:
            reward_team = 0 #-30
            info_all["scene_type"] = "Finish"
        elif done:
            assert (not bv_crash_flag) and (not cav_crash_flag)
            reward_team = 0 #60
            info_all["scene_type"] = "Finish"
        else:
            reward_team = 0

        return reward_team, info_all

    def _is_terminal(self):
        """
            The episode is over when:
            1. a collision occurs to CAV
            2. a collision occurs to BV
            3. when the access ramp has been passed.
        """
        # CAV crash flag and BV crash flag
        cav_crash_flag =self.vehicle.crashed
        bv_crash_flag = False
        bv_crash_index = []
        for vehicle in self.road.vehicles[1:]:
            if vehicle.crashed:
                bv_crash_flag = True
                bv_crash_index.append(self.road.vehicles.index(vehicle))
                # break
        # Range Over flag
        range_flag = (self.vehicle.position[0] >= self.EXIT_LENGTH)
        if cav_crash_flag and not bv_crash_flag:
            print("Miss home!")
        terminal = cav_crash_flag or bv_crash_flag or range_flag
        return terminal, cav_crash_flag, bv_crash_flag , range_flag, bv_crash_index

    def _make_vehicles(self, given_ini=None,
                       auto_vehicle=(0, global_val.initial_CAV_position, global_val.initial_CAV_speed)):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        given_ini: [[Lane 0], [Lane 1], [Lane 2]], Each item [Lane i] = [[x,velocity],..., [x,velocity]]
        """
        ego_vehicle = MDPVehicle(self.road, self.road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0),
                                 velocity=auto_vehicle[2])
        
        # ego_vehicle = IDMVehicle_CAV(self.road, self.road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0),
        #                          velocity=auto_vehicle[2])
        # print("aaaa", ego_vehicle.COMFORT_ACC_MAX)

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        other_vehicles_type = Pure_NDDVehicle 
        ini_data = None
        if given_ini:
            assert self.generate_vehicle_mode == "Given_ini"
        if self.generate_vehicle_mode == "Test":
            if len(self.presum_list_forward) == 0 or len(self.presum_list_backward) == 0:
                # If it is the first time generate by NDD, preprocess the data first
                self._preprocess_CF_data()
                print("================Generate CF data presum_list finished!================")
            lane_list = [0, 1, 2, 2]#[1, 1, 1, 2, 2, 2]  # [1, 1, 1, 2, 2, 2]
            position_list = [412, 430, 420, 440]#, 105, 120, 90, 105, 120]  # [90, 105, 120, 90, 105, 120]
            velocity_list = [20, 26, 30, 30]#, 32, 28, 36, 34, 32] # [34, 32, 28, 36, 34, 32]
            for i in range(len(lane_list)):
                v = other_vehicles_type(self.road, self.road.network.get_lane(("a", "b", lane_list[i])).position(position_list[i], 0), 0, velocity_list[i])
                self.road.vehicles.append(v)

        if self.generate_vehicle_mode == "NDD":
            if len(self.presum_list_forward) == 0 or len(self.presum_list_backward) == 0:
                # If it is the first time generate by NDD, preprocess the data first
                self._preprocess_CF_data()
                print("================Generate CF data presum_list finished!================")

            vehicle_list = []  # each list in this container is for vehicles in each lane (without CAV)
            lane_num = len(self.vehicle.road.network.graph["a"]["b"])
            for lane_idx in range(lane_num):
                generate_forward = True
                generate_finish = False
                vehicle_forward_list_one_lane, vehicle_backward_list_one_lane = [], []
                if lane_idx == 0:
                    back_vehicle_speed, front_vehicle_speed = auto_vehicle[2], auto_vehicle[2]
                    back_vehicle_position, front_vehicle_position = auto_vehicle[1], auto_vehicle[1]
                else:
                    rand_speed, rand_position, _ = self._gen_NDD_veh() 
                    back_vehicle_speed, front_vehicle_speed = rand_speed, rand_speed
                    back_vehicle_position, front_vehicle_position = rand_position, rand_position
                    vehicle_forward_list_one_lane.append((rand_position, rand_speed))

                while generate_finish is False:
                    if generate_forward is True:
                        # print(back_vehicle_speed)
                        if back_vehicle_speed < global_val.v_low:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[global_val.v_low]]
                        else:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[int(back_vehicle_speed)]]

                        # decide CF or FF
                        random_number_CF = np.random.uniform()
                        if random_number_CF > self.CF_percent:  # FF
                            rand_speed, rand_position, _ = self._gen_NDD_veh()  # self._gen_one_random_veh()
                            v_generate = rand_speed
                            pos_generate = back_vehicle_position + self.ff_dis + rand_position + global_val.LENGTH

                        else:  # CF
                            random_number = np.random.uniform()
                            r_idx, rr_idx = divmod(bisect.bisect_left(presum_list, random_number), self.num_rr)
                            try:
                                r, rr = global_val.r_to_idx_dic.inverse[r_idx], global_val.rr_to_idx_dic.inverse[rr_idx]                                
                            except:
                                # print("back_vehicle_speed:", back_vehicle_speed)
                                if back_vehicle_speed >35:
                                    r, rr = 50, -2
                                else:
                                    r, rr = 50, 2

                            # Accelerated training for initialization
                            r = r - global_val.Initial_range_adjustment_AT + global_val.Initial_range_adjustment_SG
                            if r <= global_val.Initial_range_adjustment_SG:
                                r = r + global_val.r_high
                                
                            v_generate = back_vehicle_speed + rr
                            pos_generate = back_vehicle_position + r + global_val.LENGTH

                            
                        vehicle_forward_list_one_lane.append((pos_generate, v_generate))
                        back_vehicle_speed = v_generate
                        back_vehicle_position = pos_generate

                        v = other_vehicles_type(self.road, self.road.network.get_lane(('a', 'b', lane_idx)).position(pos_generate, 0), 0, v_generate)
                        self.road.vehicles.append(v)

                        if back_vehicle_position >= self.gen_length:
                            generate_forward = False
                            generate_finish = True
                vehicle_list_each_lane = vehicle_backward_list_one_lane + vehicle_forward_list_one_lane
                vehicle_list.append(vehicle_list_each_lane)

        if self.generate_vehicle_mode == "Given_ini":
            for lane_idx in range(self.max_lane + 1):
                ini_one_lane = given_ini[lane_idx]
                for i in range(len(ini_one_lane)):
                    veh_data = ini_one_lane[i]
                    x, velocity = veh_data[0], veh_data[1]
                    v = other_vehicles_type(self.road, self.road.network.get_lane(("a", "b", lane_idx)).position(x, 0), 0, velocity)
                    self.road.vehicles.append(v)   
        
        return ini_data 