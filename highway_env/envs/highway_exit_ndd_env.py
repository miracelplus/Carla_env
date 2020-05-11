from __future__ import division, print_function, absolute_import
import numpy as np
from highway_env.envs import HighwayExitEnv
import random
from functools import reduce
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, NDDVehicle, IDMVehicle_CAV
import global_val
import scipy.io
import bisect
import heapq
from collections import namedtuple
from bidict import bidict
import os
import copy


Action = namedtuple("Action", ["cav_action", "bv_action"])
Observation = namedtuple("Observation", ["cav_observation", "bv_observation"])
Reward = namedtuple("Reward", ["cav_reward", "bv_reward"])
Indicator = namedtuple("Indicator", ["cav_indicator", "bv_indicator"])

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
        "duration": 30,
        "controlled_bv_num": 1
    }

    def __init__(self, config):
        super(HighwayExitEnvNDD, self).__init__()
        if not config:
            raise NameError("No config input!")       
        self.min_distance = config["min_distance"]
        self.max_distance = config["max_distance"]
        self.min_velocity = config["min_velocity"]
        self.max_velocity = config["max_velocity"]
        self.min_lane = config["min_lane"]
        self.max_lane = config["max_lane"]
        self.delete_BV_position = config["delete_BV_position"]
        # observation range config
        self.cav_observation_range = config["cav_observation_range"]
        self.bv_observation_range = config["bv_observation_range"]
        # We control this parameter controlled bvs from the nearest CAV beginning
        self.controlled_bv_num = config["controlled_bv_num"]
        # how many bvs will the cav/bv observe
        self.cav_observation_num = config["cav_observation_num"]
        self.bv_observation_num = config["bv_observation_num"]
        # global_val.vehicle_count = self.cav_observation_num
        self.generate_vehicle_mode = config["generate_vehicle_mode"]
        # self.log_out_traj_flag = config["log_out_traj_flag"]
        self.cav_obs_vehs_list = []
        self.log_out_traj = []  # save log out info for each episode

        # Following : NDD initialize
        if self.generate_vehicle_mode == "NDD" or self.generate_vehicle_mode == "NDD_with_exposure_frequency" or self.generate_vehicle_mode == "Test" or self.generate_vehicle_mode == "Given_ini":
            self.CF_percent = global_val.CF_percent
            self.ff_dis = global_val.ff_dis  # dis for ff
            self.gen_length = global_val.gen_length  # stop generate BV after this length
            self.presum_list_forward = global_val.presum_list_forward  # [] CDF for car following forward
            self.presum_list_backward = [1]  # CDF for car following backward
            self.speed_CDF = global_val.speed_CDF  # for both CF and FF
            self.num_r, self.num_rr, self.num_v, self.num_acc = global_val.Initialization_CF_presum_array.shape
        config = self.DEFAULT_CONFIG.copy()

    def reset(self, given_ini=None):
        """
        given_ini: [[Lane 0], [Lane 1], [Lane 2]], Each item [Lane i] = [[x,velocity],..., [x,velocity]]
        """
        self._make_road()
        ini_data = self._make_vehicles(given_ini=given_ini)
        self.cav_obs_vehs_list = []
        self.log_out_traj = []        
        super(HighwayExitEnvNDD, self).reset()
        self.determine_controlled_bv()
        # self.select_controlled_bv()
        return self.observe_cav_bv(), ini_data

    def observe_cav_bv(self):
        cav_observation, cav_action_indicator = self.get_cav_observation()
        bv_observation, bv_action_indicator = self.get_bv_observation()
        observation = Observation(cav_observation=cav_observation, bv_observation=bv_observation)
        indicator = Indicator(cav_indicator=cav_action_indicator, bv_indicator=bv_action_indicator)
        return [observation, indicator]

    def determine_controlled_bv(self):
        for vehicle in self.road.vehicles[1:]:
            vehicle.controlled = False
        self.controlled_bvs = self.road.get_BV(self.vehicle, self.controlled_bv_num, self.delete_BV_position)
        for vehicle in self.controlled_bvs:
            vehicle.controlled = True
    
    def select_controlled_bv(self):
        # Haowei Added 20200130 select the bottom lane nearest vehicle
        controlled_bv = None
        for vehicle in self.road.vehicles[1:]:
            vehicle.controlled = False
            if vehicle.lane_index[2] == 2:
                if not controlled_bv:
                    controlled_bv = vehicle
                else:
                    if abs(vehicle.position[0] - 400) < abs(controlled_bv.position[0] - 400):
                        controlled_bv = vehicle
        controlled_bv.controlled = True
        self.controlled_bvs = [controlled_bv]

    
    # Get vehicle list observed given vehicle, observation_num and CAV_flag (whether the vehicle is a CAV)
    def get_single_vehicle_observation(self, vehicle, CAV_flag, preset_observation_num=None):
        Vehicle_list = [vehicle]
        action_indicator = vehicle.get_action_indicator(ndd_flag = True, safety_flag = True, CAV_flag = CAV_flag)
        assert len(action_indicator) > 5
        if CAV_flag:
            observation_range = self.cav_observation_range
            observation_num = self.cav_observation_num
            expected_vehicle_list_len = observation_num + 1
        else:
            observation_range = self.bv_observation_range
            observation_num = self.bv_observation_num
            expected_vehicle_list_len = observation_num + 1
        if preset_observation_num:
            observation_num = preset_observation_num
            expected_vehicle_list_len = observation_num + 1

        Vehicle_list += self.road.get_BV_EU(vehicle, observation_num, observation_range)
        if len(Vehicle_list) == expected_vehicle_list_len:
            return Vehicle_list, action_indicator
        vehicle_tmp = IDMVehicle(self.road, self.road.network.get_lane(("a", "b", 0)).position(0, 0), 0, 20.1)
        for _ in range(expected_vehicle_list_len - len(Vehicle_list)):  
            Vehicle_list.append(vehicle_tmp)
        assert expected_vehicle_list_len == len(Vehicle_list)
        return Vehicle_list, action_indicator

    def Vehicle_list_to_nparray_matrix(self, Vehicle_obs_list):
        array_list = []
        indicator_list = []
        for bv_obs_and_indicator in Vehicle_obs_list:
            action_indicator = bv_obs_and_indicator[1]
            bv_obs = bv_obs_and_indicator[0]
            input_matrix = np.zeros([len(bv_obs), 3])
            for i in range(len(bv_obs)):
                input_matrix[i, 0] = bv_obs[i].position[0]
                input_matrix[i, 1] = bv_obs[i].lane_index[2]
                input_matrix[i, 2] = bv_obs[i].velocity
            input_vector = input_matrix.flatten() 
            array_list.append(input_vector)
            indicator_list.append(action_indicator)
        return array_list, indicator_list

    def check_crash_duty(self, vehicle1, vehicle2):
        """
        This function get two (crashed) vehicle and return the duty of vehicle1
        """
        if vehicle1.position[0] < vehicle2.position[0]:
            if not (vehicle1.actual_action == 'LANE_LEFT' or vehicle1.actual_action == 'LANE_RIGHT'):
                if (vehicle2.actual_action == 'LANE_LEFT' or vehicle2.actual_action == 'LANE_RIGHT') and vehicle2.target_lane_index == vehicle1.lane_index:
                    return False
            else:
                if (vehicle1.actual_action == 'LANE_LEFT' and vehicle2.actual_action == 'LANE_LEFT'):
                    if vehicle1.position[1] < vehicle2.position[1]:
                        return False
                elif (vehicle1.actual_action == 'LANE_RIGHT' and vehicle2.actual_action == 'LANE_RIGHT'):
                    if vehicle1.position[1] > vehicle2.position[1]:
                        return False
            return True
        else:
            if not (vehicle2.actual_action == 'LANE_LEFT' or vehicle2.actual_action == 'LANE_RIGHT'):
                if (vehicle1.actual_action == 'LANE_LEFT' or vehicle1.actual_action == 'LANE_RIGHT') and vehicle1.target_lane_index == vehicle2.lane_index:
                    return True
            else:
                if (vehicle1.actual_action == 'LANE_LEFT' and vehicle2.actual_action == 'LANE_LEFT'):
                    if vehicle1.position[1] > vehicle2.position[1]:
                        return True
                elif (vehicle1.actual_action == 'LANE_RIGHT' and vehicle2.actual_action == 'LANE_RIGHT'):
                    if vehicle1.position[1] < vehicle2.position[1]:
                        return True
                elif (vehicle1.actual_action == 'LANE_LEFT' and vehicle2.actual_action == 'LANE_RIGHT') or (vehicle1.actual_action == 'LANE_RIGHT' and vehicle2.actual_action == 'LANE_LEFT'):
                    if vehicle1.target_lane_index == vehicle2.target_lane_index:
                        return True
                    elif (vehicle1.target_lane_index == vehicle2.lane_index) or (vehicle2.target_lane_index == vehicle1.lane_index):
                        return True
            return False

    def get_single_bv_observation(self, bv):
        # this function get the local observation of the bv
        other_vehicle, action_indicator = self.get_single_vehicle_observation(bv, CAV_flag=False)
        return [self.vehicle] + other_vehicle, action_indicator
        # return [self.vehicle, bv], action_indicator
        # return [self.vehicle] + self.get_single_vehicle_observation(bv, CAV_flag=False) + self.controlled_bvs

    def get_bv_observation(self):
        # get observation list for bv from nearest to most distance
        whole_bvs_observation = []
        for bv in self.controlled_bvs:
            bv_obs, action_indicator = self.get_single_bv_observation(bv)
            #bv_obs += self.controlled_bvs
            whole_bvs_observation.append([bv_obs, action_indicator])
        whole_bvs_observation, whole_action_indicator = self.Vehicle_list_to_nparray_matrix(whole_bvs_observation)
        return whole_bvs_observation, whole_action_indicator

    def get_cav_observation(self):
        action_indicator = self.vehicle.get_action_indicator(ndd_flag = False, safety_flag = True, CAV_flag = True)
        obs, cav_obs_vehs_list = self.observation.original_observe_acc_training(cav_obs_num=self.cav_observation_num, cav_observation_range=self.cav_observation_range)        
        self.cav_obs_vehs_list = cav_obs_vehs_list
        return obs, action_indicator
      

    def _is_terminal(self):
        """
            The episode is over when:
            1. a collision occurs to CAV
            2. a collision occurs to BV
            2. when the access ramp has been passed.
        """
        # added by Haowei Sun 20190728 check crash finish and exit success
        # exit_flag: whether the vehicle exit the highway successfully
        _from, _to, _id = self.vehicle.lane_index
        last_id = len(self.vehicle.road.network.graph[_from][_to]) - 1
        exit_flag = self.vehicle.position[0] < self.EXIT_LENGTH and _id == last_id and not self.vehicle.crashed
        # CAV crash flag and BV crash flag
        cav_crash_flag =self.vehicle.crashed
        bv_crash_flag = 0
        bv_crash_index = []
        for vehicle in self.road.vehicles[1:]:
            if vehicle.crashed:
                bv_crash_flag = 1
                if vehicle in self.controlled_bvs:
                    bv_crash_index.append(self.controlled_bvs.index(vehicle))
                # break
        # Range Over flag
        range_flag = (self.vehicle.position[0] >= self.EXIT_LENGTH)
        if cav_crash_flag and not bv_crash_flag:
            print("Miss home!")
        terminal = cav_crash_flag or bv_crash_flag or range_flag or exit_flag
        return terminal, cav_crash_flag, bv_crash_flag , exit_flag, bv_crash_index

    def _reward_cav(self, infos, terminal):
        """
            When crashed receive -1, when exit highway receive 1, else 0
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        crash_reward = 0
        exit_highway_reward = 1
        exit_fail_reward = -1
        # scene_type = infos["car0"]["scene_type"]
        scene_type = infos["scene_type"]
        if not scene_type:
            return 0
        else:
            if scene_type == "Cexit":
                return exit_highway_reward
            elif scene_type == "CinB":
                return crash_reward
            elif scene_type == "Cfail":
                return exit_fail_reward
            else:
                return 0

        # if self.vehicle.crashed:
        #     return crash_reward
        # if info:
        #     return exit_highway_reward
        # else:
        #     if terminal:
        #         return exit_fail_reward
        #     else:
        #         return 0

    # Xintao 2020-03-17
    def _get_log_out_veh_info(self, veh, CAV_flag=False):
        """
            get the info of a given vehicle's trajectory to be log out

            input: 
                veh: the vehicle needs to be logged out
            out:
                customized vehicle info

        """
        veh_id = "CAV" if CAV_flag else veh.id
        veh_info = [global_val.episode, self.time, veh_id, veh.position[0], veh.position[1], veh.lane_index[2], veh.velocity, veh.heading, veh.weight, veh.criticality, veh.decomposed_controlled_flag]
        return veh_info

    # Xintao 2020-03-17
    def _get_log_out_veh_action(self, veh, crash_flag=False):
        """
            get the action info of a given vehicle's trajectory to be log out

            input: 
                veh: the vehicle needs to be logged out
                crash_flag: if it is crash at this timestamp, the action of each vehicle will set as None     
            output:
                higher_level_action: Left/Right/Acc
                acc
                steering angle

        """
        action = [None, None, None]
        if crash_flag:
            return action
        else:
            if veh.lane_index != veh.target_lane_index:
                if veh.lane_index[2] > veh.target_lane_index[2]: action[0] = "Left"
                elif veh.lane_index[2] < veh.target_lane_index[2]: action[0] = "Right"
            else:
                action[0] = veh.longi_acc   
        action[1], action[2] = veh.action['acceleration'], veh.action['steering']
        return action 


    def _simulate(self, cav_action=None, bv_action=None):
        """
            Perform several steps of simulation with constant action
        """
        weight_one_step = 1
        log_out_traj_freq = 1
        for k in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            essential_flag = self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])

            # get veh traj except action
            log_out_this_step_flag = self.time % log_out_traj_freq
            vehs_traj, vehs_action = [], []
            if self.log_out_traj_flag and log_out_this_step_flag == 0:
                # CAV
                veh = self.vehicle
                vehs_traj.append(self._get_log_out_veh_info(veh, CAV_flag=True))
                
                # BV
                for veh in self.cav_obs_vehs_list:
                    vehs_traj.append(self._get_log_out_veh_info(veh, CAV_flag=False))

            if ((cav_action is not None) or (bv_action is not None)) and essential_flag == 0:
                # Set the CAV and BV action
                if cav_action is not None:
                    self.vehicle.act(self.ACTIONS[cav_action], essential_flag=essential_flag)
                # If BV is controlled:
                if len(self.controlled_bvs):
                    for i in range(len(self.controlled_bvs)):
                        bv = self.controlled_bvs[i]
                        if type(bv_action[i]) == int :
                            _, ndd_possi, critical_possi = bv.act(self.BV_ACTIONS[bv_action[i]],essential_flag)
                            if critical_possi and ndd_possi:   
                                weight_tmp = ndd_possi/critical_possi
                                weight_one_step *= weight_tmp
                        else:
                            # bv.controlled = False
                            _, ndd_possi, critical_possi = bv.act(None,essential_flag)
            #  when nothing happens, vehicle act nothing
            # if cav_action is None:
            self.vehicle.act(essential_flag=essential_flag)
            self.road.act(essential_flag)
            self.road.step(1 / self.SIMULATION_FREQUENCY)
                
            # get veh action
            if self.log_out_traj_flag and log_out_this_step_flag == 0:
                # CAV
                veh = self.vehicle
                vehs_action.append(self._get_log_out_veh_action(veh, crash_flag=False))

                # BV                
                for veh in self.cav_obs_vehs_list:
                    vehs_action.append(self._get_log_out_veh_action(veh, crash_flag=False))
                
                # Combine veh traj and action and save in log_out_traj to be logged out when this episode end
                self.log_out_traj.extend([info[0]+info[1] for info in zip(vehs_traj, vehs_action)])

            self.time += 1            
            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()
            road_crash_flag = False
            for vehicle in self.road.vehicles:
                if vehicle.crashed:
                    road_crash_flag = True
                    break
            # if road_crash_flag:
            #     # Log out trajectory at the crash moment
            #     if self.log_out_traj_flag:
            #         vehs_traj, vehs_action = [], []
            #         # CAV
            #         veh = self.vehicle
            #         vehs_traj.append(self._get_log_out_veh_info(veh, CAV_flag=True))
            #         vehs_action.append(self._get_log_out_veh_action(veh, crash_flag=True))

            #         # BV                   
            #         for veh in self.cav_obs_vehs_list:
            #             vehs_traj.append(self._get_log_out_veh_info(veh, CAV_flag=False))
            #             vehs_action.append(self._get_log_out_veh_action(veh, crash_flag=True))
                
            #         # Combine veh traj and action and save in log_out_traj to be logged out when this episode end
            #         self.log_out_traj.extend([info[0]+info[1] for info in zip(vehs_traj, vehs_action)])                    

            if road_crash_flag:
                break
            
        # print("\n")
        new_vehicles_list = [self.vehicle]
        for vehicle in self.road.vehicles[1:]:
            if not (vehicle.position[0] > self.delete_BV_position) and vehicle is not self.vehicle:
                new_vehicles_list.append(vehicle)
        self.road.vehicles = new_vehicles_list
        self.enable_auto_render = False
        return weight_one_step


    def _gen_one_random_veh(self, speed_range_low=28, speed_range_high=32, pos_low=0, pos_high=50):
        """
        Generate speed and pos for one random veh
        """
        rand_speed = int(np.random.uniform(speed_range_low, speed_range_high))
        rand_position = int(np.random.uniform(pos_low, pos_high))
        return rand_speed, rand_position

    def _gen_NDD_veh(self, pos_low=global_val.random_veh_pos_buffer_start, pos_high=global_val.random_veh_pos_buffer_end):
        """
        Generate a veh. Speed by NDD, pos is random
        """
        random_number = np.random.uniform()
        idx = bisect.bisect_left(self.speed_CDF, random_number)
        exposure_freq_speed = self.speed_CDF[idx] - self.speed_CDF[idx-1] if idx >= 1 else self.speed_CDF[idx]
        speed = global_val.v_to_idx_dic.inverse[idx]
        # speed = self.road.np_random.uniform(global_val.random_initialization_BV_v_min, global_val.random_initialization_BV_v_max)  # !!!
        rand_position = round(np.random.uniform(pos_low, pos_high))
        exposure_freq = exposure_freq_speed * 1/(pos_high-pos_low)
        # print(random_number, idx, speed)
        return speed, rand_position, exposure_freq

    def _make_vehicles(self, given_ini=None,
                       auto_vehicle=(0, global_val.initial_CAV_position, global_val.initial_CAV_speed)):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        given_ini: [[Lane 0], [Lane 1], [Lane 2]], Each item [Lane i] = [[x,velocity],..., [x,velocity]]
        """
        ego_vehicle = MDPVehicle(self.road, self.road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0),
                                 velocity=auto_vehicle[2])
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        # other_vehicles_type = NDD_heuristic_Vehicle
        # other_vehicles_type = IDMVehicle
        # other_vehicles_type = IDM_heuristic_Vehicle
        other_vehicles_type = NDDVehicle 
        # other_vehicles_type = NDD_heuristic_Vehicle
        # # !!!IDMVehicle NDDVehicle
        ini_data = None
        if given_ini:
            assert self.generate_vehicle_mode == "Given_ini"
        if self.generate_vehicle_mode == "Test":
            if len(self.presum_list_forward) == 0 or len(self.presum_list_backward) == 0:
                # If it is the first time generate by NDD, preprocess the data first
                self._preprocess_CF_data()
                print("================Generate CF data presum_list finished!================")
            lane_list = [1,2,1,1,2,2,2]#[1, 1, 1, 2, 2, 2]  # [1, 1, 1, 2, 2, 2]
            position_list = [398,398,380,346,341,488,302]#, 105, 120, 90, 105, 120]  # [90, 105, 120, 90, 105, 120]
            velocity_list = [28,21,31,27,24,27,28]#, 32, 28, 36, 34, 32] # [34, 32, 28, 36, 34, 32]
            for i in range(len(lane_list)):
                v = other_vehicles_type(self.road, self.road.network.get_lane(("a", "b", lane_list[i])).position(position_list[i], 0), 0, velocity_list[i])
                self.road.vehicles.append(v)

        if self.generate_vehicle_mode == "Random":
            for _ in range(self.config["vehicles_count"]):
                new_vehicle = self.create_random_acc_training(other_vehicles_type)
                if new_vehicle:
                    self.road.vehicles.append(new_vehicle)

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
                # print(vehicle_list_each_lane)
            #     print('Lane idx:', str(lane_idx+1), " ", vehicle_list_each_lane, len(vehicle_list_each_lane))
            # print("\n")
    
        if self.generate_vehicle_mode == "NDD_with_exposure_frequency":
            if len(self.presum_list_forward) == 0 or len(self.presum_list_backward) == 0:
                # If it is the first time generate by NDD, preprocess the data first
                self._preprocess_CF_data()
                print("================Generate CF data presum_list finished!================")

            vehicle_list = []  # each list in this container is for vehicles in each lane (without CAV)
            lane_num = len(self.vehicle.road.network.graph["a"]["b"])
            exposure_freq_list = []  # exposure frequency for each lines initialization
            vehicle_data_list = [] 
            for lane_idx in range(lane_num):
                generate_forward = True
                generate_finish = False
                vehicle_forward_list_one_lane, vehicle_backward_list_one_lane = [], []
                exposure_freq_list_one_lane, vehicle_data_list_one_lane = [], []
                exposure_freq_one_lane = 1
                if lane_idx == 0:
                    back_vehicle_speed, front_vehicle_speed = auto_vehicle[2], auto_vehicle[2]
                    back_vehicle_position, front_vehicle_position = auto_vehicle[1], auto_vehicle[1]
                else:
                    rand_speed, rand_position, exposure_freq = self._gen_NDD_veh() 
                    back_vehicle_speed, front_vehicle_speed = rand_speed, rand_speed
                    back_vehicle_position, front_vehicle_position = rand_position, rand_position
                    if global_val.critical_ini_start <= back_vehicle_position <= global_val.critical_ini_end :
                        exposure_freq_one_lane *= exposure_freq
                        exposure_freq_list_one_lane.append(exposure_freq)
                        vehicle_data_list_one_lane.append([None, None, None]) # For the first BV
                    vehicle_forward_list_one_lane.append((rand_position, rand_speed))

                while generate_finish is False:
                    if generate_forward is True:
                        # print(back_vehicle_speed)
                        if back_vehicle_speed < global_val.v_low:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[global_val.v_low]]
                        else:
                            back_vehicle_speed = int(back_vehicle_speed)
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[back_vehicle_speed]]

                        # decide CF or FF
                        random_number_CF = np.random.uniform()
                        if random_number_CF > self.CF_percent or back_vehicle_speed == 40:  # FF
                            rand_speed, rand_position, exposure_freq = self._gen_NDD_veh()  # self._gen_one_random_veh()
                            v_generate = rand_speed
                            pos_generate = back_vehicle_position + self.ff_dis + rand_position + global_val.LENGTH

                            if global_val.critical_ini_start <= pos_generate <= global_val.critical_ini_end :
                                exposure_freq_one_lane *= exposure_freq
                                exposure_freq_list_one_lane.append(exposure_freq)
                                vehicle_data_list_one_lane.append([back_vehicle_speed, self.ff_dis + rand_position + global_val.LENGTH, rand_speed-back_vehicle_speed]) # !!! Need discuss

                        else:  # CF
                            random_number = np.random.uniform()
                            idx = bisect.bisect_left(presum_list, random_number)
                            r_idx, rr_idx = divmod(idx, self.num_rr)
                           
                            try:
                                r, rr = global_val.r_to_idx_dic.inverse[r_idx], global_val.rr_to_idx_dic.inverse[rr_idx]
                                                             
                            except:
                                print("back_vehicle_speed:", back_vehicle_speed) 
                                if back_vehicle_speed >35:
                                    # back_vehicle_speed = 39
                                    # presum_list = self.presum_list_forward[global_val.v_to_idx_dic[back_vehicle_speed]]
                                    
                                    # random_number = np.random.uniform()
                                    # idx = bisect.bisect_left(presum_list, random_number)
                                    # r_idx, rr_idx = divmod(idx, self.num_rr)
                                    # r, rr = global_val.r_to_idx_dic.inverse[r_idx], global_val.rr_to_idx_dic.inverse[rr_idx]  
                                    r, rr = 50, -2
                                else:
                                    r, rr = 50, 2
                                # assert("Back_vehicle_speed problem")
                                
                            v_generate = back_vehicle_speed + rr
                            pos_generate = back_vehicle_position + r + global_val.LENGTH
                            if global_val.critical_ini_start <= pos_generate <= global_val.critical_ini_end :           
                                exposure_freq = presum_list[idx] - presum_list[idx-1] if idx >= 1 else presum_list[idx] 
                                exposure_freq_one_lane *= exposure_freq
                                exposure_freq_list_one_lane.append(exposure_freq)
                                vehicle_data_list_one_lane.append([back_vehicle_speed, r + global_val.LENGTH, rr])                                 
                                                              
                        back_vehicle_speed = v_generate
                        back_vehicle_position = pos_generate

                        if back_vehicle_position >= self.gen_length:
                            generate_forward = False
                            generate_finish = True
                            continue
                        vehicle_forward_list_one_lane.append((pos_generate, v_generate))  
                        v = other_vehicles_type(self.road, self.road.network.get_lane(('a', 'b', lane_idx)).position(pos_generate, 0), 0, v_generate)
                        self.road.vehicles.append(v)


                vehicle_list_each_lane = vehicle_backward_list_one_lane + vehicle_forward_list_one_lane
                vehicle_list.append(vehicle_list_each_lane)
                # exposure_freq_list.append(exposure_freq_one_lane)
                exposure_freq_list.append(exposure_freq_list_one_lane)
                vehicle_data_list.append(vehicle_data_list_one_lane)
                # print(vehicle_list_each_lane)
            #     print('Lane idx:', str(lane_idx+1), " ", vehicle_list_each_lane, len(vehicle_list_each_lane))
            # print("\n")
            ini_data = {"data": vehicle_data_list, "exposure_frequency": exposure_freq_list, "All_initial_info": vehicle_list}

        if self.generate_vehicle_mode == "Given_ini":
            for lane_idx in range(self.max_lane + 1):
                ini_one_lane = given_ini[lane_idx]
                for i in range(len(ini_one_lane)):
                    veh_data = ini_one_lane[i]
                    x, velocity = veh_data[0], veh_data[1]
                    v = other_vehicles_type(self.road, self.road.network.get_lane(("a", "b", lane_idx)).position(x, 0), 0, velocity)
                    self.road.vehicles.append(v)   
        
        return ini_data 

    def _preprocess_CF_data(self):
        # raw_data = global_val.CF_presum_array
        raw_data = global_val.Initialization_CF_presum_array
        self.num_r, self.num_rr, self.num_v, self.num_acc = raw_data.shape
                       
        # =============================================================
        # Generate forward normalized data
        normalized_data = np.zeros(raw_data.shape[:3], dtype=float)
        for k in range(self.num_v):
            sum_prob = float(raw_data[:, :, k, :].sum())
            if sum_prob != 0:
                squeeze_acc_dim = np.sum(raw_data[:, :, k, :], axis=2)
                normalized_data[:, :, k] = squeeze_acc_dim / sum_prob
            else:
                normalized_data[:, :, k] = np.zeros(normalized_data[:, :, k].shape)
        # =============================================================
        # Construct presum list: each list is for a velocity
        for k in range(self.num_v):
            prob_vec = normalized_data[:, :, k].ravel()
            presum_vec = [round(sum(prob_vec[:idx+1]), 6) if round(sum(prob_vec[:idx+1]), 6) < 0.999999 else 1.0 for idx in range(len(prob_vec))]
            self.presum_list_forward.append(presum_vec)

    def create_random_acc_training(self, other_vehicles_type, velocity=None):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :return: A vehicle with random position and/or velocity
        """
        if velocity is None:
            velocity = self.road.np_random.uniform(global_val.random_initialization_BV_v_min, global_val.random_initialization_BV_v_max)
        _from = self.road.np_random.choice(list(self.road.network.graph.keys()))
        _to = self.road.np_random.choice(list(self.road.network.graph[_from].keys()))
        num = 0
        while(1):
            num += 1
            _id = self.road.np_random.choice(len(self.road.network.graph[_from][_to]))
            x0 = np.random.uniform(global_val.random_env_BV_generation_range_min, global_val.random_env_BV_generation_range_max)  # self.HIGHWAY_LENGTH
            check_dis_x = [abs(x0 - v.position[0]) for v in self.road.vehicles]
            check_dis_y = [abs(_id - v.lane_index[2]) for v in self.road.vehicles]
            check = [True if ele1 < self.config["minimum_distance"] and ele2 < 0.1 else False for (ele1, ele2) in \
                     zip(check_dis_x, check_dis_y)]
            if True not in check:
                break
            if num > 40:
                return None
        # print(x0, velocity)
        v = other_vehicles_type(self.road, self.road.network.get_lane((_from, _to, _id)).position(x0, 0), 0, velocity)
        return v

