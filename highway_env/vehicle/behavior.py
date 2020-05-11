from __future__ import division, print_function
import numpy as np
from highway_env.vehicle.control import ControlledVehicle
from highway_env import utils
import global_val
import bisect
import scipy.io
import os
import scipy
import copy
import scipy.stats

def cal_mixed_acc(range, range_rate):
    acc_range = 2*(range+range_rate)
    acc_range_rate = range_rate
    if range == 0 and range_rate == 0:
        return 0
    acc_mixed_coeff = abs(range)/(abs(range)+abs(range_rate))
    acc_mixed = acc_mixed_coeff*acc_range + (1-acc_mixed_coeff)*acc_range_rate
    return acc_mixed
    # return acc_range

class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
        """
    ACC_LIST = []
    BV_ACTIONS = {0: 'LANE_LEFT', 1: 'LANE_RIGHT'}
    num_acc = int(((global_val.acc_high - global_val.acc_low)/global_val.acc_step) + 1)
    num_non_acc = len(BV_ACTIONS)
    for i in range(num_acc):
        acc = global_val.acc_to_idx_dic.inverse[i]
        BV_ACTIONS[i+num_non_acc] = str(acc)  
        ACC_LIST.append(acc)
    ACC_LIST = np.array(ACC_LIST)
    LANE_CHANGE_INDEX_LIST = [0, 1, 2] 
    NDD_ACC_MIN = global_val.acc_low
    NDD_ACC_MAX = global_val.acc_high
    # Longitudinal policy parameters
    COMFORT_ACC_MAX = 2.0  # [m/s2]  2
    COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
    DISTANCE_WANTED = 5.0  # [m]  5
    TIME_WANTED = 1.5  # [s]  1.5
    DESIRED_VELOCITY = 35 # [m/s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]  0
    LANE_CHANGE_MIN_ACC_GAIN = 0.  # [m/s2]  0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0  # [m/s2]  2
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.IDM_flag = False
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        self.action_mode = "DriverModel"  # NDD or DriverModel
        self.longi_acc = 0


    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v
    
    # 20191008 Modify Xintao
    def act(self, bv_action=None):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        if bv_action or self.controlled:
            self.IDM_flag = False
            self.follow_road()
            _from, _to, _id = self.lane_index
            # Haowei added actual action 20191125
            if bv_action:
                self.actual_action = bv_action
                if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                    self.actual_action = "IDLE"
                elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                    self.actual_action = "IDLE"
                
            if bv_action == "LANE_RIGHT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif bv_action == "LANE_LEFT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif bv_action == "IDLE":
                self.longi_acc = 0
            elif bv_action:
                self.longi_acc = int(bv_action)
            action = {'steering': self.steering_control(self.target_lane_index),
                    'acceleration': self.longi_acc}
            super(ControlledVehicle, self).act(action)
            return
        
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)        

        if self.action_mode == "DriverModel":
            self.IDM_flag = True
            # Lateral: MOBIL
            self.follow_road()
            if self.enable_lane_change:
                lane_change_flag, _ = self.change_lane_policy()
            action['steering'] = self.steering_control(self.target_lane_index)

            # Longitudinal: IDM
            #action['acceleration'] = self.acceleration(ego_vehicle=self,front_vehicle=front_vehicle,rear_vehicle=rear_vehicle)
            tmp_acc = self.acceleration(ego_vehicle=self,front_vehicle=front_vehicle,rear_vehicle=rear_vehicle)
            tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
            acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
            acc_possi_list = acc_possi_list/(sum(acc_possi_list))
            #self.longi_acc = np.random.normal(, 0.2, None)
            action['acceleration'] = np.random.choice(self.ACC_LIST,None,False,acc_possi_list)
            super(ControlledVehicle, self).act(action)
            self.longi_acc = action['acceleration']
            return action['acceleration'],acc_possi_list

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(IDMVehicle, self).step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        # acceleration = self.COMFORT_ACC_MAX * (
        #         1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), self.DELTA))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / self.DESIRED_VELOCITY, self.DELTA))                
        if front_vehicle:
            d = max(1e-5, ego_vehicle.lane_distance_to(front_vehicle) - self.LENGTH)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED 
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self, modify_flag=True):
        """
            Decide when to change lane.

            Based on:
            - frequency;
            - closeness of the target lane;
            - MOBIL model.

            When modify_flag is False, it just predict the lane change decision of the CAV and not do the real control of the CAV
        """
        to_lane_id = None

        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            #print("self_index:",self.lane_index)
                            #print("change_index:",lane_index)
                            self.target_lane_index = self.lane_index
                            break
            return False, to_lane_id

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return False, to_lane_id
        if modify_flag:
            self.timer = 0
        
        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                #print("self_index:",self.lane_index)
                #print("change_index:",lane_index)
                if modify_flag:
                    self.target_lane_index = lane_index
                to_lane_id = lane_index[2]
                return True, to_lane_id
        return False, to_lane_id

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change
            决定要不要换道，并不决定具体的加速度等
            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
            if jerk <= self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration):
        """
            If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_velocity = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

class IDMVehicle_CAV(IDMVehicle):
    ACC_LIST = []
    BV_ACTIONS = {0: 'LANE_LEFT', 1: 'LANE_RIGHT'}
    num_acc = int(((global_val.acc_high - global_val.acc_low)/global_val.acc_step) + 1)
    num_non_acc = len(BV_ACTIONS)
    for i in range(num_acc):
        acc = global_val.acc_to_idx_dic.inverse[i]
        BV_ACTIONS[i+num_non_acc] = str(acc)  
        ACC_LIST.append(acc)
    ACC_LIST = np.array(ACC_LIST)
    LANE_CHANGE_INDEX_LIST = [0, 1, 2] 
    NDD_ACC_MIN = global_val.acc_low
    NDD_ACC_MAX = global_val.acc_high
    # Longitudinal policy parameters
    COMFORT_ACC_MAX = 2.0  # [m/s2]
    COMFORT_ACC_MIN = -2.0  # [m/s2]
    DISTANCE_WANTED = 2.0  # [m]
    TIME_WANTED = 1.  # [s]
    DESIRED_VELOCITY = 35 # [m/s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.5  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
    
    mode, v, x, lane_idx, surrounding_vehs = None, None, None, None, None
    
    mode_prev, longi_acc_prev = None, None
    v_prev, x_prev, lane_idx_prev, surrounding_vehs_prev = None, None, None, None

    def _get_info_of_surrounding_vehs(self, obs):
        surrounding_vehs_info = []
        for veh in obs:
            if veh:
                surrounding_vehs_info.append((veh.velocity, veh.position[0], veh.lane_index[2]))
            else:
                surrounding_vehs_info.append((None, None, None))
        return surrounding_vehs_info
                


    def act(self, bv_action=None, essential_flag = False):
        if self.crashed:
            return
        action = {}
        if essential_flag == 0:
            if bv_action or self.controlled:
                self.IDM_flag = False
                self.follow_road()
                _from, _to, _id = self.lane_index
                # Haowei added actual action 20191125
                if bv_action:
                    self.actual_action = bv_action
                    if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                        self.actual_action = "IDLE"
                    elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                        self.actual_action = "IDLE"
                    
                if bv_action == "LANE_RIGHT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                elif bv_action == "LANE_LEFT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                elif bv_action == "IDLE":
                    self.longi_acc = 0
                elif bv_action:
                    self.longi_acc = int(bv_action)
                action = {'steering': self.steering_control(self.target_lane_index),
                        'acceleration': self.longi_acc}
                super(ControlledVehicle, self).act(action)
                return

            self.mode_prev, self.longi_acc_prev, self.v_prev, self.x_prev, self.lane_idx_prev, self.surrounding_vehs_prev = copy.deepcopy(self.mode), copy.deepcopy(self.longi_acc), copy.deepcopy(self.v), copy.deepcopy(self.x), copy.deepcopy(self.lane_idx), copy.deepcopy(self.surrounding_vehs)

            obs = self._get_obs_for_safety_check()
            self.surrounding_vehs = self._get_info_of_surrounding_vehs(obs)
            self.v, self.x, self.lane_idx = copy.deepcopy(self.velocity), copy.deepcopy(self.position[0]), copy.deepcopy(self.lane_index[2])

            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)        

            if self.action_mode == "DriverModel":
                self.IDM_flag = True
                # Lateral: MOBIL
                self.mode = "MOBIL"
                self.follow_road()
                if self.enable_lane_change:
                    lane_change_flag, _ = self.change_lane_policy()
                action['steering'] = self.steering_control(self.target_lane_index)

                # Longitudinal: IDM
                if not lane_change_flag:
                    self.mode = "IDM"
                    # tmp_acc = self.acceleration(ego_vehicle=self,front_vehicle=front_vehicle,rear_vehicle=rear_vehicle)
                    # tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
                    # acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
                    # acc_possi_list = acc_possi_list/(sum(acc_possi_list))
                    # #self.longi_acc = np.random.normal(, 0.2, None)
                    # action['acceleration'] = np.random.choice(self.ACC_LIST,None,False,acc_possi_list)
                    tmp_acc = self.acceleration(ego_vehicle=self,front_vehicle=front_vehicle,rear_vehicle=rear_vehicle)
                    tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
                    action['acceleration'] = tmp_acc
                    super(ControlledVehicle, self).act(action)
                    self.longi_acc = action['acceleration']
                    # return action['acceleration'],acc_possi_list
                    return action['acceleration'], None
                else:
                    self.longi_acc = 0

        action['acceleration'] = self.longi_acc
        action['steering'] = self.steering_control(self.target_lane_index)
        super(ControlledVehicle, self).act(action)

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change
            决定要不要换道，并不决定具体的加速度等
            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        
        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding.position[0] - self.position[0] - self.LENGTH
        if new_following:
            r_new_following = self.position[0] - new_following.position[0] - self.LENGTH            
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False

        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            gain = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
            if gain <= self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

class NDDVehicle(IDMVehicle):

    def get_action_indicator(self, ndd_flag = False, safety_flag = True, CAV_flag = False):
        """
        This get the observation and return the possibility bounded by safety guard
        """
        if CAV_flag:
            action_shape = len(global_val.ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_obs_for_safety_check()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=True)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape-2), CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]
                # exp_action_Q_full[0], exp_action_Q_full[1] = 0, 0        
                safety_action_indicator[2:] = longi_result                
            action_indicator = ndd_action_indicator * safety_action_indicator
            return action_indicator
        else:
            action_shape = len(global_val.BV_ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                longi_possi_list = self.get_Longi_NDD_possi_list()
                lateral_possi_list = self.get_Lateral_NDD_possi_list()
                ndd_action_indicator[0] = lateral_possi_list[0]
                ndd_action_indicator[1] = lateral_possi_list[2]
                ndd_action_indicator[2:] = longi_possi_list * lateral_possi_list[1]
            # ndd_action_indicator = (ndd_action_indicator > 0)
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_obs_for_safety_check()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=False)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape-2), CAV_flag=False)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2] 
                safety_action_indicator[2:] = longi_result   
            safety_action_indicator = (safety_action_indicator > 0) 
            action_indicator = ndd_action_indicator * safety_action_indicator
            assert sum((action_indicator < 0)) == 0
            action_indicator = action_indicator/np.sum(action_indicator)
            # action_indicator = (action_indicator > 0)
            return action_indicator

    def get_Lateral_NDD_possi_list(self):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        lane_id = self.lane_index[2] 
        observation = []  #  obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None                    
            observation = [f1, r1, f0, r0, f2, r2]
        _, _, lane_change_pdf_array = self.Lateral_NDD(observation, modify_flag=False)
        return lane_change_pdf_array     

    def get_Longi_NDD_possi_list(self):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        lane_id = self.lane_index[2] 
        observation = []  #  obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None                    
            observation = [f1, r1, f0, r0, f2, r2]
        _, possi_list = self.Longitudinal_NDD(observation)
        return possi_list
    
    def act(self, bv_action=None, essential_flag = False):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        action = {}
        if self.crashed:
            return
        if essential_flag == 0:
            self.IDM_flag = False # Reset the veh flag to NDD veh flag
            lane_id = self.lane_index[2] 
            observation = self._get_obs_for_safety_check()
            if bv_action:
                self.IDM_flag = False
                self.follow_road()
                _from, _to, _id = self.lane_index
                # Haowei added actual action 20191125
                if bv_action:
                    self.actual_action = bv_action
                    if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                        self.actual_action = "IDLE"
                    elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                        self.actual_action = "IDLE"
                    
                if bv_action == "LANE_RIGHT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                elif bv_action == "LANE_LEFT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                elif bv_action == "IDLE":
                    self.longi_acc = 0
                elif bv_action:
                    self.longi_acc = float(bv_action)
                _, _, lane_change_pdf_array = self.Lateral_NDD(observation, modify_flag=False)
                _, longi_pdf_array = self.Longitudinal_NDD(observation)
                action = {'steering': self.steering_control(self.target_lane_index), 'acceleration': self.longi_acc}
                super(ControlledVehicle, self).act(action)
                ndd_possi = 0
                if self.actual_action == "LANE_LEFT":
                    ndd_possi = lane_change_pdf_array[0]
                elif self.actual_action == "LANE_RIGHT":
                    ndd_possi = lane_change_pdf_array[2]
                else:
                    acc_idx = list(self.ACC_LIST).index(self.longi_acc)             
                    ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1]   
                return action, ndd_possi, None
            
            else:
                # Lateral: NDD
                self.follow_road()
                lane_change_flag = False
                if self.enable_lane_change:
                    lane_change_flag, lane_change_idx, lane_change_pdf_array = self.Lateral_NDD(observation)
                    # print(lane_change_pdf_array)

                # Longitudinal: NDD
                if not lane_change_flag:
                    self.longi_acc, longi_pdf_array = self.Longitudinal_NDD(observation)
                else:
                    _, longi_pdf_array = self.Longitudinal_NDD(observation)
                    self.longi_acc = 0
            
        action['acceleration'] = self.longi_acc
        action['steering'] = self.steering_control(self.target_lane_index)
        super(ControlledVehicle, self).act(action)
        if essential_flag == 0:
            ndd_possi = 0
            if lane_change_idx == 0:
                ndd_possi = lane_change_pdf_array[0]
            elif lane_change_idx == 2:
                ndd_possi = lane_change_pdf_array[2]
            else:
                acc_idx = list(self.ACC_LIST).index(self.longi_acc)   
                ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1] 
        else:
            ndd_possi = None
        return action, ndd_possi,None

    # Round value for lane change
    def round_value_lane_change(self, real_value, value_list, round_item="speed"):
        if real_value < value_list[0]: real_value = value_list[0]
        elif real_value > value_list[-1]: real_value = value_list[-1]        
        
        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            return round_value, value_idx
        else: 
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            return round_value, value_idx
        # if global_val.round_rule == "Round up":
        #     value_idx = bisect.bisect_left(value_list, real_value) 
        #     value_idx = value_idx if value_list[value_idx]==real_value or value_list[value_idx] == value_list[-1] else value_idx + 1
        #     try:
        #         assert value_idx <= (len(value_list)-1)
        #     except:
        #         print("Fxxk!!!")
        #     round_value = value_list[value_idx]
        #     return round_value, value_idx

    def _check_bound_constraints(self, value, bound_low, bound_high):
        if value < bound_low or value > bound_high:
            return False
        else:
            return True

    def _Lane_change_decision_with_vehicle_adjacent(self, surrounding_vehicles):
        left_front_v, left_back_v, right_front_v, right_back_v = surrounding_vehicles
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        speed_list, rf_list, re_list, rrf_list, rre_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))

        adj_back_v = None
        if left_front_v:
            adj_front_v = left_front_v
            if left_back_v:
                adj_back_v = left_back_v
        else:
            adj_front_v = right_front_v
            if right_back_v:
                adj_back_v = right_back_v

        # Double lane change
        if adj_back_v:
            speed = adj_back_v.velocity
            rf = adj_front_v.position[0] - adj_back_v.position[0]
            rrf = adj_front_v.velocity - adj_back_v.velocity
            re = x - adj_back_v.position[0]
            rre = v - adj_back_v.velocity
            if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
                # print("Speed of BV:",str(speed), ", exit [",str(global_val.lc_v_low),",",str(global_val.lc_v_high),"]! (Double lane)");
                return False
            elif not self._check_bound_constraints(rf, global_val.lc_rf_low, global_val.lc_rf_high): 
                # print("rf of BV:",str(rf), ", exit [",str(global_val.lc_rf_low),",",str(global_val.lc_rf_high),"]! (Double lane)"); 
                return False
            elif not self._check_bound_constraints(rrf, global_val.lc_rrf_low, global_val.lc_rrf_high): 
                # print("rrf of BV:",str(rrf), ", exit [",str(global_val.lc_rrf_low),",",str(global_val.lc_rrf_high),"]! (Double lane)"); 
                return False               
            elif not self._check_bound_constraints(re, global_val.lc_re_low, global_val.lc_re_high): 
                # print("re of BV:",str(re), ", exit [",str(global_val.lc_re_low),",",str(global_val.lc_re_high),"]! (Double lane)"); 
                return False
            elif not self._check_bound_constraints(rre, global_val.lc_rre_low, global_val.lc_rre_high): 
                # print("rre of BV:",str(rre), ", exit [",str(global_val.lc_rre_low),",",str(global_val.lc_rre_high),"]! (Double lane)"); 
                return False
            speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list,round_item="speed")
            rf, rf_idx = self.round_value_lane_change(real_value=rf, value_list=rf_list)
            rrf, rrf_idx = self.round_value_lane_change(real_value=rrf, value_list=rrf_list)
            re, re_idx = self.round_value_lane_change(real_value=re, value_list=re_list)
            rre, rre_idx = self.round_value_lane_change(real_value=rre, value_list=rre_list)
            
            lane_change_prob = global_val.DLC_pdf[speed_idx,rf_idx,rrf_idx,re_idx,rre_idx,:][0]
            return lane_change_prob

        # Single lane change
        else:
            speed = v
            r = adj_front_v.position[0] - x - self.LENGTH
            rr = adj_front_v.velocity - v
            if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
                # print("Speed of BV:",str(speed), ", exit [",str(global_val.lc_v_low),",",str(global_val.lc_v_high),"]! (Single lane change)"); 
                return False           
            elif not self._check_bound_constraints(r, global_val.lc_re_low, global_val.lc_re_high): 
                # print("re of BV:",str(r), ", exit [",str(global_val.lc_re_low),",",str(global_val.lc_re_high),"]! (Single lane change)"); 
                return False
            elif not self._check_bound_constraints(rr, global_val.lc_rre_low, global_val.lc_rre_high): 
                # print("rre of BV:",str(rr), ", exit [",str(global_val.lc_rre_low),",",str(global_val.lc_rre_high),"]! (Single lane change)"); 
                return False
            
            speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list, round_item="speed")
            r, r_idx = self.round_value_lane_change(real_value=r, value_list=re_list)
            rr, rr_idx = self.round_value_lane_change(real_value=rr, value_list=rre_list)
            lane_change_prob = global_val.SLC_pdf[speed_idx,r_idx,rr_idx,:][0]
            return lane_change_prob
                
    # 20191008 Add Xintao
    def Lateral_NDD(self, obs, modify_flag=True):
        """
        Decide the Lateral movement
        Input: observation of surrounding vehicles
        Output: whether do lane change (True, False), lane_change_idx (0:Left, 1:Still, 2:Right), action_pdf
        """
        initial_pdf = np.array([0,1,0])  # Left, Still, Right
        if not list(global_val.OL_pdf):
            # self._process_lane_change_data()
            assert("No OL_pdf file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs
        
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")
        # Lane change
        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehcile ahead
            # One lead
            if not f0 and not f2:  # No vehicle both adjacent
                r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                # Check bound
                if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                    # print("One lead out bound!!!!!!", v, r, rr)
                    return False, 1, initial_pdf

                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                # presum_array = global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx]
                # presum_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
                # presum_array[1] = 1 - (1-presum_array[1]) * global_val.LANE_CHANGE_SCALING_FACTOR
                # lane_change_idx = bisect.bisect_left(presum_array, np.random.uniform())

                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
                pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
                pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
                pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]
                pdf_array = self._check_lateral_safety(obs, pdf_array)            
                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                
                if lane_change_idx != 1:
                    to_lane_id = lane_id + lane_change_idx - 1
                    if to_lane_id >= 0 and to_lane_id <= 2:
                        if modify_flag:
                            self.target_lane_index = ("a", "b", to_lane_id)
                        return True, lane_change_idx, pdf_array
                    else:
                        # !!! NEEDS ATTENTION!!! 
                        if lane_id == 0:
                            pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
                            pdf_array[0] = 0
                            pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
                            pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]            
                            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                             
                            return False, lane_change_idx, pdf_array
                        elif lane_id == 2:
                            pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
                            pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
                            pdf_array[2] = 0
                            pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]            
                            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                             
                            return False, lane_change_idx, pdf_array                            
                else:
                    return False, lane_change_idx, pdf_array 
            
            # Has adjecent vehicle
            else:
                pdf_array = np.zeros((3))  # Initial Left, Still, Right
                # Not in the middle lane
                if lane_id != 1:
                    if f2: # In the leftest lane
                        right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        left_prob, stright_prob = 0., 1-right_prob
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        pdf_array = self._check_lateral_safety(obs, pdf_array)            
                        lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                
                        if lane_change_idx == 2:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 1)
                            return True, lane_change_idx, pdf_array
                        else:
                            return False, lane_change_idx, pdf_array
                    if f0: # In the rightest lane
                        left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        right_prob, stright_prob = 0., 1-left_prob
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        pdf_array = self._check_lateral_safety(obs, pdf_array)                  
                        lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                                        
                        if lane_change_idx == 0:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 1)
                            return True, lane_change_idx, pdf_array
                        else:
                            return False, lane_change_idx, pdf_array
                
                # In the middle lane
                else: 
                    if f0 and f2: # both side have vehicles
                        left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        if left_prob + right_prob > 1:
                            # print("=============Left+Right>1")
                            tmp = left_prob + right_prob
                            left_prob *= 0.9/(tmp)
                            right_prob *= 0.9/(tmp)
                        stright_prob = 1-right_prob-left_prob
                        # presum_array = [left_prob, left_prob+stright_prob, left_prob+stright_prob+right_prob]
                        # lane_change_idx = bisect.bisect_left(presum_array, np.random.uniform())
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        pdf_array = self._check_lateral_safety(obs, pdf_array)                 
                        try:
                            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)         
                        except:
                            print(pdf_array)               
                        if lane_change_idx == 0:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 0)
                            return True, lane_change_idx, pdf_array
                        if lane_change_idx == 1:
                            return False, lane_change_idx, pdf_array
                        if lane_change_idx == 2:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 2)
                            return True, lane_change_idx, pdf_array
                    else:  # One side is empty
                        if f0: # Left has vehicle
                            left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                            right_prob = 0
                            # One lead change
                            r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                            # Check bound
                            if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                                # print("=============One lead out bound!!!!!!")
                                right_prob = 0
                            else:
                                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                                # presum_array = global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx]
                                # right_prob = (presum_array[2] - presum_array[1]) * global_val.LANE_CHANGE_SCALING_FACTOR
                                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])  
                                right_prob = pdf_array[2] * global_val.LANE_CHANGE_SCALING_FACTOR         
                            if left_prob + right_prob > 1:
                                # print("=============Left+Right>1")
                                tmp = left_prob + right_prob
                                left_prob *= 0.9/(tmp)
                                right_prob *= 0.9/(tmp)
                            stright_prob = 1-right_prob-left_prob
                            # presum_array = [left_prob, left_prob+stright_prob, left_prob+stright_prob+right_prob]
                            # lane_change_idx = bisect.bisect_left(presum_array, np.random.uniform())
                            pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                            pdf_array = self._check_lateral_safety(obs, pdf_array)     
                            try:
                                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                             
                            except:
                                print(pdf_array)
                            if lane_change_idx == 0:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", 0)
                                return True, lane_change_idx, pdf_array
                            if lane_change_idx == 1:
                                return False, lane_change_idx, pdf_array
                            if lane_change_idx == 2:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", 2)
                                return True, lane_change_idx, pdf_array
                        else:
                            right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                            left_prob = 0
                            # One lead change
                            r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                            # Check bound
                            if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                                # print("=============One lead out bound!!!!!!")
                                left_prob = 0
                            else:
                                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                                # presum_array = global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx]
                                # left_prob = presum_array[0] * global_val.LANE_CHANGE_SCALING_FACTOR
                                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])  
                                left_prob = pdf_array[0] * global_val.LANE_CHANGE_SCALING_FACTOR   
                            if left_prob + right_prob > 1:
                                # print("=============Left+Right>1")
                                tmp = left_prob + right_prob        
                                left_prob *= 0.9/(tmp)
                                right_prob *= 0.9/(tmp)
                            stright_prob = 1-right_prob-left_prob
                            # presum_array = [left_prob, left_prob+stright_prob, left_prob+stright_prob+right_prob]
                            # lane_change_idx = bisect.bisect_left(presum_array, np.random.uniform())
                            pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                            pdf_array = self._check_lateral_safety(obs, pdf_array)            
                            try:
                                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)
                            except:
                                print(pdf_array) 
                            if lane_change_idx == 0:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", 0)
                                return True, lane_change_idx, pdf_array
                            if lane_change_idx == 1:
                                return False, lane_change_idx, pdf_array
                            if lane_change_idx == 2:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", 2)
                                return True, lane_change_idx, pdf_array

    # 20191204 Add Xintao round the speed according to specific rule
    def round_value_function(self, real_value, round_item):
        if round_item == "speed":
            value_list = global_val.speed_list
            value_dic = global_val.v_to_idx_dic
        elif round_item == "range":
            value_list = global_val.r_list
            value_dic = global_val.r_to_idx_dic
        elif round_item == "range_rate":
            value_list = global_val.rr_list
            value_dic = global_val.rr_to_idx_dic
        
        if real_value < value_list[0]: real_value = value_list[0]
        elif real_value > value_list[-1]: real_value = value_list[-1]

        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            assert value_dic[round_value] == value_idx
            return round_value, value_idx
        else: 
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            assert value_dic[round_value] == value_idx
            return round_value, value_idx
        # if global_val.round_rule == "Round up":
        #     value_idx = bisect.bisect_left(value_list, real_value) 
        #     value_idx = value_idx if value_list[value_idx]==real_value or value_list[value_idx] == value_list[-1] else value_idx + 1
        #     try:
        #         assert value_idx <= (len(value_list)-1)
        #     except:
        #         print("Fxxk!!!")
        #     round_value = value_list[value_idx]
        #     assert value_dic[round_value] == value_idx
        #     return round_value, value_idx

    # 20191008 Add Xintao
    def Longitudinal_NDD(self, obs):
        """
        Decide the Longitudinal acceleration
        Input: observation of surrounding vehicles
        Output: Acceleration
        """
        if not list(global_val.CF_pdf_array):
            # self._process_CF_data()
            assert("No CF_pdf_array file!")
        if not list(global_val.FF_pdf_array):
            # self._process_FF_data()
            assert("No FF_pdf_array file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs
        
        # If r < given threshold, then change to IDM/MOBIL directly
        if f1: 
            r = f1.position[0] - x - self.LENGTH
            if r < global_val.r_threshold_NDD:
                # print("r<", global_val.r_threshold_NDD)
                pdf_array = self.stochastic_IDM()
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                return acc, pdf_array

        if not f1:  # No vehicle ahead. Then FF
            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            pdf_array = global_val.FF_pdf_array[round_speed_idx]   
            pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
            acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
            return acc, pdf_array
        
        else:  # Has vehcile ahead. Then CF
            r = f1.position[0] - x - self.LENGTH
            rr = f1.velocity - v
            if not self._check_bound_constraints(r, global_val.r_low, global_val.r_high) or not self._check_bound_constraints(rr, global_val.rr_low, global_val.rr_high) or not self._check_bound_constraints(v, global_val.v_low, global_val.v_high):
                pdf_array = self.stochastic_IDM()
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                return acc, pdf_array

            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            round_r, round_r_idx = self.round_value_function(r, round_item="range")
            round_rr, round_rr_idx = self.round_value_function(rr, round_item="range_rate")

            pdf_array = global_val.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
            # if sum(presum_array) == 0:
            if sum(pdf_array) == 0:
                # print("No CF data", round_speed, round_r, round_rr)
                pdf_array = self.stochastic_IDM()
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                return acc, pdf_array
            # acc = global_val.acc_to_idx_dic.inverse[acc_idx]
            pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
            acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
            return acc, pdf_array        
            # return acc, pdf_array

    def stochastic_IDM(self):
        self.IDM_flag = True
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        tmp_acc = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
        acc_possi_list = acc_possi_list/(sum(acc_possi_list))
        #self.longi_acc = np.random.normal(, 0.2, None)
        # acc = np.random.choice(self.ACC_LIST,None,False,acc_possi_list)
        
        return acc_possi_list


class Pure_NDDVehicle(IDMVehicle):

    mode = None # In this timestamp, this veh is doing CF or FF or LC or IDM
    v, x, lane_idx, r, rr = None, None, None, None, None # The v, r, rr of the vehicle at the specific timestamp when doing the decision
    round_r, round_rr, round_v = None, None, None
    pdf_distribution, ndd_possi = None, None # In this timestamp, the action pdf distribution and the probability of choosing the current action
    LC_related = None

    mode_prev = None # In this timestamp, this veh is doing CF or FF or LC or IDM
    v_prev, x_prev, lane_idx_prev, r_prev, rr_prev = None, None, None, None, None # The v, r, rr of the vehicle at the specific timestamp when doing the decision
    round_r_prev, round_rr_prev, round_v_prev = None, None, None
    pdf_distribution_prev, ndd_possi_prev = None, None # In this timestamp, the action pdf distribution and the probability of choosing the current action
    longi_acc_prev = None
    LC_related_prev = None

    def get_action_indicator(self, ndd_flag = False, safety_flag = True, CAV_flag = False):
        """
        This get the observation and return the possibility bounded by safety guard
        """
        if CAV_flag:
            action_shape = len(global_val.ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_obs_for_safety_check()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=True)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape-2), CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]
                # exp_action_Q_full[0], exp_action_Q_full[1] = 0, 0        
                safety_action_indicator[2:] = longi_result                
            action_indicator = ndd_action_indicator * safety_action_indicator
            return action_indicator
        else:
            action_shape = len(global_val.BV_ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                longi_possi_list = self.get_Longi_NDD_possi_list()
                lateral_possi_list = self.get_Lateral_NDD_possi_list()
                ndd_action_indicator[0] = lateral_possi_list[0]
                ndd_action_indicator[1] = lateral_possi_list[2]
                ndd_action_indicator[2:] = longi_possi_list * lateral_possi_list[1]
            # ndd_action_indicator = (ndd_action_indicator > 0)
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_obs_for_safety_check()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=False)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape-2), CAV_flag=False)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2] 
                safety_action_indicator[2:] = longi_result   
            safety_action_indicator = (safety_action_indicator > 0) 
            action_indicator = ndd_action_indicator * safety_action_indicator
            assert sum((action_indicator < 0)) == 0
            action_indicator = action_indicator/np.sum(action_indicator)
            # action_indicator = (action_indicator > 0)
            return action_indicator

    def get_Lateral_NDD_possi_list(self):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        lane_id = self.lane_index[2] 
        observation = []  #  obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None                    
            observation = [f1, r1, f0, r0, f2, r2]
        _, _, lane_change_pdf_array = self.Lateral_NDD_New(observation, modify_flag=False)
        return lane_change_pdf_array     

    def get_Longi_NDD_possi_list(self):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        lane_id = self.lane_index[2] 
        observation = []  #  obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None                    
            observation = [f1, r1, f0, r0, f2, r2]
        _, possi_list = self.Longitudinal_NDD(observation)
        return possi_list
    
    def act(self, bv_action=None, essential_flag = False):
        """
            Execute an action.

        :param action: the action
               essential_flag: whether this timestamp decide the action. 0 for sample a new action, other for not
        """
        action = {}
        if self.crashed:
            return

        if essential_flag == 0:
            # if self.mode != "IDM":
            self.mode_prev, self.pdf_distribution_prev = copy.deepcopy(self.mode), copy.deepcopy(self.pdf_distribution)
            self.v_prev, self.r_prev, self.rr_prev, self.round_v_prev, self.round_r_prev, self.round_rr_prev, self.ndd_possi_prev, self.longi_acc_prev = copy.deepcopy(self.v), copy.deepcopy(self.r), copy.deepcopy(self.rr), copy.deepcopy(self.round_v), copy.deepcopy(self.round_r), copy.deepcopy(self.round_rr), copy.deepcopy(self.ndd_possi), copy.deepcopy(self.longi_acc) 
            self.LC_related_prev = copy.deepcopy(self.LC_related)
            self.x_prev, self.lane_idx_prev = copy.deepcopy(self.x), copy.deepcopy(self.lane_idx)

            self.mode, self.pdf_distribution = None, None
            self.v, self.x, self.lane_idx = copy.deepcopy(self.velocity), copy.deepcopy(self.position[0]), copy.deepcopy(self.lane_index[2])            
            self.r, self.rr, self.round_v, self.round_r, self.round_rr, self.ndd_possi, self.LC_related =  None, None, None, None, None, None, None            
            
            self.IDM_flag = False # Reset the veh flag to NDD veh flag
            lane_id = self.lane_index[2] 
            observation = self._get_obs_for_safety_check()
            if bv_action:
                self.follow_road()
                _from, _to, _id = self.lane_index
                if bv_action:
                    self.actual_action = bv_action
                    if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                        self.actual_action = "IDLE"
                        self.mode = "Controlled-Should not here!"
                    elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                        self.actual_action = "IDLE"
                        self.mode = "Controlled-Should not here!"
                    
                if bv_action == "LANE_RIGHT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                    self.mode = "Controlled-Right"
                elif bv_action == "LANE_LEFT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                    self.mode = "Controlled-Left"
                elif bv_action == "IDLE":
                    self.longi_acc = 0
                    self.mode = "Controlled-IDLE"
                elif bv_action:
                    self.mode = "Controlled-Long"
                    self.longi_acc = float(bv_action)
                _, _, lane_change_pdf_array = self.Lateral_NDD_New(observation, modify_flag=False)
                _, longi_pdf_array = self.Longitudinal_NDD(observation)
                action = {'steering': self.steering_control(self.target_lane_index), 'acceleration': self.longi_acc}
                super(ControlledVehicle, self).act(action)
                ndd_possi = 0
                if self.actual_action == "LANE_LEFT":
                    ndd_possi = lane_change_pdf_array[0]
                elif self.actual_action == "LANE_RIGHT":
                    ndd_possi = lane_change_pdf_array[2]
                else:
                    acc_idx = list(self.ACC_LIST).index(self.longi_acc)             
                    ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1]   
                return action, ndd_possi, None
            
            else:
                # Lateral: NDD
                self.follow_road()
                lane_change_flag = False
                if self.enable_lane_change:
                    # lane_change_flag, lane_change_idx, lane_change_pdf_array = self.Lateral_NDD(observation)
                    lane_change_flag, lane_change_idx, lane_change_pdf_array = self.Lateral_NDD_New(observation)                    
                    # print(lane_change_pdf_array)
                if not global_val.enable_One_lead_LC and not global_val.enable_Single_LC and not global_val.enable_Double_LC:
                    assert (not lane_change_flag)
                # Longitudinal: NDD
                if not lane_change_flag:
                    self.longi_acc, longi_pdf_array = self.Longitudinal_NDD(observation, modify_flag=True)
                    higher_action = self.longi_acc
                else:
                    # _, longi_pdf_array = self.Longitudinal_NDD(observation)
                    if self.target_lane_index[2] == self.lane_index[2] - 1:
                        higher_action = "Left"
                    elif self.target_lane_index[2] == self.lane_index[2] + 1:
                        higher_action = "Right"
                    self.longi_acc = 0
            
        action['acceleration'] = self.longi_acc
        action['steering'] = self.steering_control(self.target_lane_index)
        super(ControlledVehicle, self).act(action)
        if essential_flag == 0:
            ndd_possi = 0
            if lane_change_idx == 0:
                ndd_possi = lane_change_pdf_array[0]
            elif lane_change_idx == 2:
                ndd_possi = lane_change_pdf_array[2]
            else:
                acc_idx = list(self.ACC_LIST).index(self.longi_acc)   
                ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1] 
            self.ndd_possi = ndd_possi
        else:
            ndd_possi = None
        return action, ndd_possi, higher_action

    # Round value for lane change
    def round_value_lane_change(self, real_value, value_list, round_item=None):
        if real_value < value_list[0]: real_value = value_list[0]
        elif real_value > value_list[-1]: real_value = value_list[-1]

        if global_val.round_rule == "Round_to_closest":
            min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
            real_value = np.clip(round((real_value - (min_val)) / resolution)*resolution + (min_val), min_val, max_val)

        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            assert value_idx <= (len(value_list)-1)
            assert value_idx >= 0
            round_value = value_list[value_idx]
            return round_value, value_idx
        else: 
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
            assert value_idx <= (len(value_list)-1)
            assert value_idx >= 0
            round_value = value_list[value_idx]
            return round_value, value_idx

    def _check_bound_constraints(self, value, bound_low, bound_high):
        if value < bound_low or value > bound_high:
            return False
        else:
            return True

    def _Lane_change_decision_with_vehicle_adjacent(self, surrounding_vehicles):
        left_front_v, left_back_v, right_front_v, right_back_v = surrounding_vehicles
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        speed_list, rf_list, re_list, rrf_list, rre_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))

        adj_back_v = None
        if left_front_v:
            adj_front_v = left_front_v
            if left_back_v:
                adj_back_v = left_back_v
        else:
            adj_front_v = right_front_v
            if right_back_v:
                adj_back_v = right_back_v

        # Double lane change
        if adj_back_v:
            if not global_val.enable_Double_LC:
                return 0            
            speed = adj_back_v.velocity
            # speed = v # !!!!!!!!!!
            rf = adj_front_v.position[0] - adj_back_v.position[0] - self.LENGTH
            rrf = adj_front_v.velocity - adj_back_v.velocity
            re = x - adj_back_v.position[0] - self.LENGTH
            rre = v - adj_back_v.velocity
            if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
                # print("Speed of BV:",str(speed), ", exit [",str(global_val.lc_v_low),",",str(global_val.lc_v_high),"]! (Double lane)");
                return False
            elif not self._check_bound_constraints(rf, global_val.lc_rf_low, global_val.lc_rf_high): 
                # print("rf of BV:",str(rf), ", exit [",str(global_val.lc_rf_low),",",str(global_val.lc_rf_high),"]! (Double lane)"); 
                return False
            elif not self._check_bound_constraints(rrf, global_val.lc_rrf_low, global_val.lc_rrf_high): 
                # print("rrf of BV:",str(rrf), ", exit [",str(global_val.lc_rrf_low),",",str(global_val.lc_rrf_high),"]! (Double lane)"); 
                return False               
            elif not self._check_bound_constraints(re, global_val.lc_re_low, global_val.lc_re_high): 
                # print("re of BV:",str(re), ", exit [",str(global_val.lc_re_low),",",str(global_val.lc_re_high),"]! (Double lane)"); 
                return False
            elif not self._check_bound_constraints(rre, global_val.lc_rre_low, global_val.lc_rre_high): 
                # print("rre of BV:",str(rre), ", exit [",str(global_val.lc_rre_low),",",str(global_val.lc_rre_high),"]! (Double lane)"); 
                return False
            round_speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list,round_item="speed")
            round_rf, rf_idx = self.round_value_lane_change(real_value=rf, value_list=rf_list)
            round_rrf, rrf_idx = self.round_value_lane_change(real_value=rrf, value_list=rrf_list)
            round_re, re_idx = self.round_value_lane_change(real_value=re, value_list=re_list)
            round_rre, rre_idx = self.round_value_lane_change(real_value=rre, value_list=rre_list)
            
            lane_change_prob = global_val.DLC_pdf[speed_idx,rf_idx,rrf_idx,re_idx,rre_idx,:][0]
            # if lane_change_prob >0:
            #     print("lane_change_prob:", lane_change_prob)

            self.LC_related = (speed, rf, rrf, re, rre, round_speed, round_rf, round_rrf, round_re, round_rre, lane_change_prob)  
            return lane_change_prob

        # Single lane change
        else:
            if not global_val.enable_Single_LC:
                return 0
            speed = v
            r = adj_front_v.position[0] - x - self.LENGTH
            rr = adj_front_v.velocity - v
            if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
                # print("Speed of BV:",str(speed), ", exit [",str(global_val.lc_v_low),",",str(global_val.lc_v_high),"]! (Single lane change)"); 
                return False           
            elif not self._check_bound_constraints(r, global_val.lc_re_low, global_val.lc_re_high): 
                # print("re of BV:",str(r), ", exit [",str(global_val.lc_re_low),",",str(global_val.lc_re_high),"]! (Single lane change)"); 
                return False
            elif not self._check_bound_constraints(rr, global_val.lc_rre_low, global_val.lc_rre_high): 
                # print("rre of BV:",str(rr), ", exit [",str(global_val.lc_rre_low),",",str(global_val.lc_rre_high),"]! (Single lane change)"); 
                return False
            
            speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list, round_item="speed")
            round_r, r_idx = self.round_value_lane_change(real_value=r, value_list=re_list)
            round_rr, rr_idx = self.round_value_lane_change(real_value=rr, value_list=rre_list)
            assert(round_r <= r and round_rr <= rr)
            lane_change_prob = global_val.SLC_pdf[speed_idx,r_idx,rr_idx,:][0]
            return lane_change_prob

    def _get_One_lead_LC_prob(self, veh_front, full_obs):
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        if not global_val.enable_One_lead_LC:
            return 0, None                   
        r, rr = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        # Check bound
        if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
            # print("One lead out bound!!!!!!", v, r, rr)
            return 0, None

        round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
        round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")
        assert((round_r-1) <= r and r <= (round_r+1) and (round_rr-1) <= rr and rr <= (round_rr+1))

        pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
        pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
        pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
        pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]

        if global_val.safety_guard_enabled_flag:          
            pdf_array = self._check_lateral_safety(full_obs, pdf_array)
        One_lead_prob = 0.5* (pdf_array[0]+pdf_array[2])
        LC_related = (v, r, rr, round_speed, round_r, round_rr,pdf_array)            
        return One_lead_prob, LC_related       
        
    def _get_Double_LC_prob(self, veh_adj_front, veh_adj_rear, full_obs):              
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        speed_list, rf_list, re_list, rrf_list, rre_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))
        LC_related = None
        # Double lane change
        if not global_val.enable_Double_LC:
            return 0, LC_related    
        adj_front_v, adj_back_v = veh_adj_front, veh_adj_rear            
        speed = adj_back_v.velocity
        # speed = v # !!!!!!!!!!
        rf = adj_front_v.position[0] - adj_back_v.position[0] - self.LENGTH
        rrf = adj_front_v.velocity - adj_back_v.velocity
        re = x - adj_back_v.position[0] - self.LENGTH
        rre = v - adj_back_v.velocity
        if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rf, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rrf, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related             
        elif not self._check_bound_constraints(re, global_val.lc_re_low, global_val.lc_re_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rre, global_val.lc_rre_low, global_val.lc_rre_high): 
            return 0, LC_related
        round_speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list,round_item="speed")
        round_rf, rf_idx = self.round_value_lane_change(real_value=rf, value_list=rf_list)
        round_rrf, rrf_idx = self.round_value_lane_change(real_value=rrf, value_list=rrf_list)
        round_re, re_idx = self.round_value_lane_change(real_value=re, value_list=re_list)
        round_rre, rre_idx = self.round_value_lane_change(real_value=rre, value_list=rre_list)
        
        lane_change_prob = global_val.DLC_pdf[speed_idx,rf_idx,rrf_idx,re_idx,rre_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        LC_related = (speed, rf, rrf, re, rre, round_speed, round_rf, round_rrf, round_re, round_rre, lane_change_prob)  
        return lane_change_prob, LC_related
    
    def _get_Single_LC_prob(self, veh_front, veh_adj_front, full_obs):              
        # lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        # speed_list, rf_list, rrf_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num))

        # LC_related = None
        # # Single lane change
        # if not global_val.enable_Single_LC:
        #     return 0, LC_related    
        
        # speed = v
        # rf = veh_adj_front.position[0] - x - self.LENGTH
        # rrf = veh_adj_front.velocity - speed

        # if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
        #     return 0, LC_related
        # elif not self._check_bound_constraints(rf, global_val.lc_rf_low, global_val.lc_rf_high): 
        #     return 0, LC_related
        # elif not self._check_bound_constraints(rrf, global_val.lc_rrf_low, global_val.lc_rrf_high): 
        #     return 0, LC_related             

        # round_speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list,round_item="speed")
        # round_rf, rf_idx = self.round_value_lane_change(real_value=rf, value_list=rf_list)
        # round_rrf, rrf_idx = self.round_value_lane_change(real_value=rrf, value_list=rrf_list)
        
        # lane_change_prob = global_val.SLC_pdf[speed_idx,rf_idx,rrf_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        # LC_related = (speed, rf, rrf, round_speed, round_rf, round_rrf, lane_change_prob)  
        # return lane_change_prob, LC_related

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        speed_list, rf_list, r_adj_list, rrf_list, rr_adj_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))

        LC_related = None
        # Single lane change
        if not global_val.enable_Single_LC:
            return 0, LC_related    
        
        speed = v
        rf = veh_front.position[0] - x - self.LENGTH
        rrf = veh_front.velocity - speed
        r_adj = veh_adj_front.position[0] - x - self.LENGTH
        rr_adj =  veh_adj_front.velocity - speed

        if not self._check_bound_constraints(speed, global_val.lc_v_low, global_val.lc_v_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rf, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rrf, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related             
        elif not self._check_bound_constraints(r_adj, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related 
        elif not self._check_bound_constraints(rr_adj, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related 

        round_speed, speed_idx = self.round_value_lane_change(real_value=speed, value_list=speed_list,round_item="speed")
        round_rf, rf_idx = self.round_value_lane_change(real_value=rf, value_list=rf_list)
        round_rrf, rrf_idx = self.round_value_lane_change(real_value=rrf, value_list=rrf_list)
        round_r_adj, r_adj_idx = self.round_value_lane_change(real_value=r_adj, value_list=r_adj_list)
        round_rr_adj, rr_adj_idx = self.round_value_lane_change(real_value=rr_adj, value_list=rr_adj_list)        
        
        lane_change_prob = global_val.SLC_pdf[speed_idx,rf_idx,rrf_idx,r_adj_idx,rr_adj_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        LC_related = (speed, rf, rrf, r_adj, rr_adj, round_speed, round_rf, round_rrf, round_r_adj, round_rr_adj, lane_change_prob)  
        if lane_change_prob >0 :
            print("OOOOLLLL", lane_change_prob, LC_related)
        return lane_change_prob, LC_related
        

    def _get_Cut_in_LC_prob(self, veh_front, veh_adj_rear, full_obs):
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        if not global_val.enable_Cut_in_LC:
            return 0, None                   
        r, rr = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        r_adj_rear, rr_adj_rear = x - veh_adj_rear.position[0] - self.LENGTH, v - veh_adj_rear.velocity
        if (r_adj_rear >= self.LENGTH) and (r_adj_rear > global_val.Cut_in_veh_adj_rear_threshold - rr_adj_rear * 1):
            # Check bound
            if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                return 0, None

            round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
            round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
            round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")
            assert((round_r-1) <= r and r <= (round_r+1) and (round_rr-1) <= rr and rr <= (round_rr+1))

            pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
            pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
            pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
            pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]

            if global_val.safety_guard_enabled_flag:          
                pdf_array = self._check_lateral_safety(full_obs, pdf_array)
            Cut_in_LC_prob = 0.5* (pdf_array[0]+pdf_array[2])
            LC_related = (v, r, rr, r_adj_rear, rr_adj_rear, round_speed, round_r, round_rr,pdf_array)            
            return Cut_in_LC_prob, LC_related   
        else:
            return 0, None

    def _LC_prob(self, surrounding_vehicles, full_obs):
        """
        Input: (veh_front, veh_adj_front, veh_adj_back)
        output: the lane change probability
        """
        LC_prob = None
        veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles
        if not veh_adj_front and not veh_adj_rear:
            # One lead LC
            LC_prob, LC_related = self._get_One_lead_LC_prob(veh_front, full_obs)
            return LC_prob, "One_lead", LC_related
        
        elif veh_adj_front and not veh_adj_rear:
            # Single lane change
            LC_prob, LC_related = self._get_Single_LC_prob(veh_front, veh_adj_front, full_obs)
            return LC_prob, "SLC", LC_related

        elif not veh_adj_front and veh_adj_rear:
            # Cut in
            # Now no data
            LC_prob, LC_related = self._get_Cut_in_LC_prob(veh_front, veh_adj_rear, full_obs)
            return LC_prob, "Cut_in", LC_related

        elif veh_adj_front and veh_adj_rear:
            # Double lane change
            LC_prob, LC_related = self._get_Double_LC_prob(veh_adj_front, veh_adj_rear, full_obs)
            return LC_prob, "DLC", LC_related


    # 20200221 Add Xintao
    def Lateral_NDD_New(self, obs, modify_flag=True):
        """
        Decide the Lateral movement
        Input: observation of surrounding vehicles
        Output: whether do lane change (True, False), lane_change_idx (0:Left, 1:Still, 2:Right), action_pdf
        """
        initial_pdf = np.array([0,1,0])  # Left, Still, Right
        if not list(global_val.OL_pdf):
            # self._process_lane_change_data()
            assert("No OL_pdf file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs
        
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")

        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehcile ahead
            left_prob, still_prob, right_prob = 0, 0, 0
            LC_related_list = []
            LC_type_list = []
            for item in ["Left", "Right"]:
                if item == "Left":
                    surrounding = (f1, f0, r0)
                    left_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
                else:
                    surrounding = (f1, f2, r2)
                    right_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)

            # In the leftest or rightest
            if lane_id == 0:
                left_prob = 0
            if lane_id == 2:
                right_prob = 0
            if left_prob + right_prob > 1:
                tmp = left_prob + right_prob        
                left_prob *= 0.9/(tmp)
                right_prob *= 0.9/(tmp)                
            still_prob = 1 - left_prob - right_prob
            pdf_array = np.array([left_prob, still_prob, right_prob])
            if global_val.safety_guard_enabled_flag:          
                pdf_array = self._check_lateral_safety(obs, pdf_array)                  
            
            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)    
            to_lane_id = lane_id + lane_change_idx - 1
            if lane_change_idx != 1:
                if modify_flag:
                    self.target_lane_index = ("a", "b", to_lane_id)
                    if lane_change_idx == 0:
                        self.mode = LC_type_list[0]
                        self.LC_related = LC_related_list[0]
                        # print("LLC", self.mode, self.LC_related, pdf_array)
                    elif lane_change_idx == 2:
                        self.mode = LC_type_list[1]
                        self.LC_related = LC_related_list[1]
                        # print("RLC", self.mode, self.LC_related, pdf_array)
                return True, lane_change_idx, pdf_array
            else:
                return False, lane_change_idx, pdf_array

    # 20191008 Add Xintao
    def Lateral_NDD(self, obs, modify_flag=True):
        """
        Decide the Lateral movement
        Input: observation of surrounding vehicles
        Output: whether do lane change (True, False), lane_change_idx (0:Left, 1:Still, 2:Right), action_pdf
        """
        initial_pdf = np.array([0,1,0])  # Left, Still, Right
        if not list(global_val.OL_pdf):
            # self._process_lane_change_data()
            assert("No OL_pdf file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs
        
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")
        # Lane change
        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehcile ahead
            # One lead        
            if not f0 and not f2:  # No vehicle both adjacent
                if not global_val.enable_One_lead_LC:
                    return False, 1, initial_pdf                   
                r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                # Check bound
                if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                    # print("One lead out bound!!!!!!", v, r, rr)
                    return False, 1, initial_pdf

                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                # assert(round_r <= r and round_rr <= rr)

                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
                pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
                pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
                pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]
                if lane_id == 0:
                    pdf_array[0] = 0
                    pdf_array[1] = 1 - pdf_array[2]
                elif lane_id == 2:
                    pdf_array[2] = 0
                    pdf_array[1] = 1 - pdf_array[0]          
                if global_val.safety_guard_enabled_flag:          
                    pdf_array = self._check_lateral_safety(obs, pdf_array)
                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,size=None,replace=False,p=pdf_array)                
                if lane_change_idx != 1:
                    to_lane_id = lane_id + lane_change_idx - 1
                    assert(0<=lane_change_idx<=2)
                    # if to_lane_id >= 0 and to_lane_id <= 2:
                    if modify_flag:
                        self.target_lane_index = ("a", "b", to_lane_id)
                    self.mode = "One_lead_LC"
                    self.LC_related = (lane_change_idx, v, r, rr, round_speed, round_r, round_rr, pdf_array)                       
                    return True, lane_change_idx, pdf_array                         
                else:
                    return False, lane_change_idx, pdf_array 
            
            # Has adjecent vehicle
            else:
                pdf_array = np.zeros((3))  # Initial Left, Still, Right
                # Not in the middle lane
                if lane_id != 1:
                    if f2: # In the leftest lane
                        right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        left_prob, stright_prob = 0., 1-right_prob
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        if global_val.safety_guard_enabled_flag:          
                            pdf_array = self._check_lateral_safety(obs, pdf_array)
                        lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                
                        if lane_change_idx == 2:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 1)
                            self.mode = "Double_LC"                               
                            return True, lane_change_idx, pdf_array
                        else:
                            return False, lane_change_idx, pdf_array
                    if f0: # In the rightest lane
                        left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        right_prob, stright_prob = 0., 1-left_prob
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        if global_val.safety_guard_enabled_flag:          
                            pdf_array = self._check_lateral_safety(obs, pdf_array)                  
                        lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                                        
                        if lane_change_idx == 0:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 1)
                            return True, lane_change_idx, pdf_array
                        else:
                            return False, lane_change_idx, pdf_array
                
                # In the middle lane
                else: 
                    if f0 and f2: # both side have vehicles
                        left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                        if left_prob + right_prob > 1:
                            # print("=============Left+Right>1")
                            tmp = left_prob + right_prob
                            left_prob *= 0.9/(tmp)
                            right_prob *= 0.9/(tmp)
                        stright_prob = 1-right_prob-left_prob
                        pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                        if global_val.safety_guard_enabled_flag:          
                            pdf_array = self._check_lateral_safety(obs, pdf_array)                 
                        try:
                            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)         
                        except:
                            print(pdf_array)               
                        if lane_change_idx == 0:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 0)
                            self.mode = "Double_LC"          
                            return True, lane_change_idx, pdf_array
                        if lane_change_idx == 1:
                            return False, lane_change_idx, pdf_array
                        if lane_change_idx == 2:
                            if modify_flag:
                                self.target_lane_index = ("a", "b", 2)
                            self.mode = "Double_LC"
                            return True, lane_change_idx, pdf_array
                    else:  # One side is empty
                        if f0: # Left has vehicle
                            left_prob = self._Lane_change_decision_with_vehicle_adjacent((f0,r0,None,None)) * global_val.LANE_CHANGE_SCALING_FACTOR
                            right_prob = 0
                            # One lead change
                            r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                            # Check bound
                            if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                                # print("=============One lead out bound!!!!!!")
                                right_prob = 0
                            else:
                                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                                # presum_array = global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx]
                                # right_prob = (presum_array[2] - presum_array[1]) * global_val.LANE_CHANGE_SCALING_FACTOR
                                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])  
                                right_prob = pdf_array[2] * global_val.LANE_CHANGE_SCALING_FACTOR     
                                if not global_val.enable_One_lead_LC:
                                    right_prob = 0

                            if left_prob + right_prob > 1:
                                # print("=============Left+Right>1")
                                tmp = left_prob + right_prob
                                left_prob *= 0.9/(tmp)
                                right_prob *= 0.9/(tmp)
                            stright_prob = 1-right_prob-left_prob
                            pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                            if global_val.safety_guard_enabled_flag:          
                                pdf_array = self._check_lateral_safety(obs, pdf_array)      
                            try:
                                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)                             
                            except:
                                print(pdf_array)
                            to_lane_id = lane_id + lane_change_idx - 1
                            if lane_change_idx != 1:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", to_lane_id)   
                                self.mode = "Double_LC"
                                return True, lane_change_idx, pdf_array
                            else:
                                return False, lane_change_idx, pdf_array
                        else:
                            right_prob = self._Lane_change_decision_with_vehicle_adjacent((None,None,f2,r2)) * global_val.LANE_CHANGE_SCALING_FACTOR
                            left_prob = 0
                            # One lead change
                            r, rr = f1.position[0] - x - self.LENGTH, f1.velocity - v
                            # Check bound
                            if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
                                # print("=============One lead out bound!!!!!!")
                                left_prob = 0
                            else:
                                round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
                                round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
                                pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])  
                                left_prob = pdf_array[0] * global_val.LANE_CHANGE_SCALING_FACTOR   
                                if not global_val.enable_One_lead_LC:
                                    left_prob = 0     
                            if left_prob + right_prob > 1:
                                # print("=============Left+Right>1")
                                tmp = left_prob + right_prob        
                                left_prob *= 0.9/(tmp)
                                right_prob *= 0.9/(tmp)
                            stright_prob = 1-right_prob-left_prob
                            pdf_array[0], pdf_array[1], pdf_array[2] = left_prob, stright_prob, right_prob
                            if global_val.safety_guard_enabled_flag:          
                                pdf_array = self._check_lateral_safety(obs, pdf_array)                
                            try:
                                lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)
                            except:
                                print(pdf_array) 
                            to_lane_id = lane_id + lane_change_idx - 1
                            if lane_change_idx != 1:
                                if modify_flag:
                                    self.target_lane_index = ("a", "b", to_lane_id)
                                self.mode = "Double_LC"   
                                return True, lane_change_idx, pdf_array
                            else:
                                return False, lane_change_idx, pdf_array

    # 20191204 Add Xintao round the speed according to specific rule
    def round_value_function(self, real_value, round_item):
        if round_item == "speed":
            value_list = global_val.speed_list
            value_dic = global_val.v_to_idx_dic
            min_val, max_val, resolution = global_val.v_low, global_val.v_high, global_val.v_step
        elif round_item == "range":
            value_list = global_val.r_list
            value_dic = global_val.r_to_idx_dic
            min_val, max_val, resolution = global_val.r_low, global_val.r_high, global_val.r_step
        elif round_item == "range_rate":
            value_list = global_val.rr_list
            value_dic = global_val.rr_to_idx_dic
            min_val, max_val, resolution = global_val.rr_low, global_val.rr_high, global_val.rr_step

        if real_value < value_list[0]: real_value = value_list[0]
        elif real_value > value_list[-1]: real_value = value_list[-1]

        if global_val.round_rule == "Round_to_closest":
            real_value = np.clip(round((real_value - (min_val)) / resolution)*resolution + (min_val), min_val, max_val)
        
        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx if real_value != value_list[value_idx-1] else value_idx - 1
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            assert value_dic[round_value] == value_idx
            return round_value, value_idx
        else: 
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Fxxk!!!")
            round_value = value_list[value_idx]
            assert value_dic[round_value] == value_idx
            return round_value, value_idx

    # 20191008 Add Xintao
    def Longitudinal_NDD(self, obs, modify_flag = False):
        """
        Decide the Longitudinal acceleration
        Input: observation of surrounding vehicles
                if modify_flag is True, then is really doing action control, otherwise it just retrieving NDD possi
        Output: Acceleration
        """
        if not list(global_val.CF_pdf_array):
            # self._process_CF_data()
            assert("No CF_pdf_array file!")
        if not list(global_val.FF_pdf_array):
            # self._process_FF_data()
            assert("No FF_pdf_array file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, _, _, _, _, _ = obs
        
        # If r < given threshold, then change to IDM/MOBIL directly
        if f1: 
            r = f1.position[0] - x - self.LENGTH
            if r < global_val.r_threshold_NDD:
                # assert("Should not enter here!")   
                pdf_array = self.stochastic_IDM()
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                if modify_flag:
                    self.mode, self.pdf_distribution = "IDM", pdf_array  
                    self.v = v                            
                return acc, pdf_array

        if not f1:  # No vehicle ahead. Then FF
            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            pdf_array = global_val.FF_pdf_array[round_speed_idx]   
            if global_val.safety_guard_enabled_flag:
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
            acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
            if modify_flag:
                self.mode, self.v, self.round_v, self.pdf_distribution = "FF", v, round_speed, pdf_array 
            return acc, pdf_array
        
        else:  # Has vehcile ahead. Then CF
            r = f1.position[0] - x - self.LENGTH
            rr = f1.velocity - v
            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            round_r, round_r_idx = self.round_value_function(r, round_item="range")
            round_rr, round_rr_idx = self.round_value_function(rr, round_item="range_rate")            

            if not self._check_bound_constraints(r, global_val.r_low, global_val.r_high) or not self._check_bound_constraints(rr, global_val.rr_low, global_val.rr_high) or not self._check_bound_constraints(v, global_val.v_low, global_val.v_high):
                pdf_array = self.stochastic_IDM()
                if global_val.safety_guard_enabled_flag or global_val.safety_guard_enabled_flag_IDM:
                    pdf_array = self._check_longitudinal_safety(obs, pdf_array)            
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                if modify_flag:                
                    self.mode, self.pdf_distribution = "IDM", pdf_array 
                    self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr
                return acc, pdf_array

            # assert(round_r <= r and round_rr <= rr)

            pdf_array = global_val.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
            if sum(pdf_array) == 0:
                # print("No CF data", round_speed, round_r, round_rr)
                pdf_array = self.stochastic_IDM()
                if global_val.safety_guard_enabled_flag or global_val.safety_guard_enabled_flag_IDM:
                    pdf_array = self._check_longitudinal_safety(obs, pdf_array)              
                acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
                if modify_flag:
                    self.mode, self.pdf_distribution = "IDM", pdf_array 
                    self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr                
                return acc, pdf_array
            if global_val.safety_guard_enabled_flag:
                pdf_array = self._check_longitudinal_safety(obs, pdf_array)              
            acc = np.random.choice(self.ACC_LIST,None,False,pdf_array)
            if modify_flag:
                self.mode, self.pdf_distribution = "CF", pdf_array 
                self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr            
            return acc, pdf_array        

    def stochastic_IDM(self):
        self.IDM_flag = True
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        tmp_acc = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
        # Delete possi if smaller than certain threshold
        acc_possi_list = [val if val > global_val.Stochastic_IDM_threshold else 0 for val in acc_possi_list]
        acc_possi_list = acc_possi_list/(sum(acc_possi_list))
        
        return acc_possi_list

# 2020/04/21 This class is for New queryed NDD data. This only difference between this one and Pure_NDDVehicle is the decision part of the vehicle
# Now retain the Pure_NDDVehicle class is to guarantee the previous results can be reproduced
class New_data_Pure_NDDVehicle(Pure_NDDVehicle):

    mode = None # In this timestamp, this veh is doing CF or FF or LC or IDM
    v, x, lane_idx, r, rr = None, None, None, None, None # The v, r, rr of the vehicle at the specific timestamp when doing the decision
    round_r, round_rr, round_v = None, None, None
    pdf_distribution, ndd_possi = None, None # In this timestamp, the action pdf distribution and the probability of choosing the current action
    LC_related = None

    mode_prev = None # In this timestamp, this veh is doing CF or FF or LC or IDM
    v_prev, x_prev, lane_idx_prev, r_prev, rr_prev = None, None, None, None, None # The v, r, rr of the vehicle at the specific timestamp when doing the decision
    round_r_prev, round_rr_prev, round_v_prev = None, None, None
    pdf_distribution_prev, ndd_possi_prev = None, None # In this timestamp, the action pdf distribution and the probability of choosing the current action
    longi_acc_prev = None
    LC_related_prev = None
    

    def _get_One_lead_LC_prob(self, veh_front, full_obs):
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        if not global_val.enable_One_lead_LC:
            return 0, None                   
        r, rr = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        # Check bound
        if not self._check_bound_constraints(v, global_val.one_lead_v_low, global_val.one_lead_v_high) or not self._check_bound_constraints(r, global_val.one_lead_r_low, global_val.one_lead_r_high) or not self._check_bound_constraints(rr, global_val.one_lead_rr_low, global_val.one_lead_rr_high):
            # print("One lead out bound!!!!!!", v, r, rr)
            return 0, None

        round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.one_lead_r_list)
        round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.one_lead_rr_list)
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")
        assert((round_r-1) <= r and r <= (round_r+1) and (round_rr-1) <= rr and rr <= (round_rr+1))

        lane_change_prob = global_val.OL_pdf[round_speed_idx,round_r_idx,round_rr_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR
        LC_related = (v, r, rr, round_speed, round_r, round_rr)     
        # pdf_array = copy.deepcopy(global_val.OL_pdf[round_r_idx, round_rr_idx, round_speed_idx])
        # pdf_array[0] *= global_val.LANE_CHANGE_SCALING_FACTOR
        # pdf_array[2] *= global_val.LANE_CHANGE_SCALING_FACTOR
        # pdf_array[1] = 1 - pdf_array[0] - pdf_array[2]

        # if global_val.safety_guard_enabled_flag:          
        #     pdf_array = self._check_lateral_safety(full_obs, pdf_array)
        # lane_change_prob = 0.5* (pdf_array[0]+pdf_array[2])
    
        # LC_related = (v, r, rr, round_speed, round_r, round_rr,pdf_array)            
        return lane_change_prob, LC_related       
        
    def _get_Double_LC_prob(self, veh_adj_front, veh_adj_rear, full_obs):              
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))
        LC_related = None
        # Double lane change
        if not global_val.enable_Double_LC:
            return 0, LC_related 
        r1, rr1 = veh_adj_front.position[0] - x - self.LENGTH, veh_adj_front.velocity - v
        r2, rr2 = x - veh_adj_rear.position[0] - self.LENGTH, v - veh_adj_rear.velocity
        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related             
        elif not self._check_bound_constraints(r2, global_val.lc_re_low, global_val.lc_re_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rr2, global_val.lc_rre_low, global_val.lc_rre_high): 
            return 0, LC_related
        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=v_list,round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=rr2_list)
        
        lane_change_prob = global_val.DLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)  
        return lane_change_prob, LC_related
    
    def _get_Single_LC_prob(self, veh_front, veh_adj_front, full_obs):              
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))

        LC_related = None
        # Single lane change
        if not global_val.enable_Single_LC:
            return 0, LC_related    
        
        r1, rr1 = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        r2, rr2 = veh_adj_front.position[0] - x - self.LENGTH, veh_adj_front.velocity - v

        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related             
        elif not self._check_bound_constraints(r2, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related 
        elif not self._check_bound_constraints(rr2, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related 

        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=v_list,round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=rr2_list)        
        
        lane_change_prob = global_val.SLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)  
        return lane_change_prob, LC_related
        

    def _get_Cut_in_LC_prob(self, veh_front, veh_adj_rear, full_obs):
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(global_val.lc_v_low,global_val.lc_v_high,num=global_val.lc_v_num)), list(np.linspace(global_val.lc_rf_low,global_val.lc_rf_high,num=global_val.lc_rf_num)), list(np.linspace(global_val.lc_re_low,global_val.lc_re_high,num=global_val.lc_re_num)), list(np.linspace(global_val.lc_rrf_low,global_val.lc_rrf_high,num=global_val.lc_rrf_num)), list(np.linspace(global_val.lc_rre_low,global_val.lc_rre_high,num=global_val.lc_rre_num))

        LC_related = None
        
        if not global_val.enable_Cut_in_LC:
            return 0, None   

        r1, rr1 = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        r2, rr2 = x - veh_adj_rear.position[0] - self.LENGTH, v - veh_adj_rear.velocity

        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related             
        elif not self._check_bound_constraints(r2, global_val.lc_rf_low, global_val.lc_rf_high): 
            return 0, LC_related 
        elif not self._check_bound_constraints(rr2, global_val.lc_rrf_low, global_val.lc_rrf_high): 
            return 0, LC_related 

        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=v_list,round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=rr2_list)        
        
        lane_change_prob = global_val.CI_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * global_val.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob) 
        return lane_change_prob, LC_related

    
    def _LC_rear_veh_check(self, ego_veh, adj_rear_veh):
        """
        Check the range for ego vehicle and the rear vehicle in the adjacent lane, if it is larger than the threshold, then has no influence 
        True: The rear vehicle has influence
        False: Does not need to consider the rear vehicle
        """
        r_adj = ego_veh.position[0] - adj_rear_veh.position[0] - ego_veh.LENGTH
        if r_adj <= global_val.LC_range_threshold:
            return True, r_adj
        else:
            return False, r_adj


    def _LC_prob(self, surrounding_vehicles, full_obs):
        """
        Input: (veh_front, veh_adj_front, veh_adj_back)
        output: the lane change probability and the expected lane change probability (take the ignored situation into account)
        """
        LC_prob, E_LC_prob = None, None
        veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles

        if not veh_adj_front and not veh_adj_rear:
            # One lead LC
            LC_prob, LC_related = self._get_One_lead_LC_prob(veh_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "One_lead", LC_related
        
        elif veh_adj_front and not veh_adj_rear:
            # Single lane change
            LC_prob, LC_related = self._get_Single_LC_prob(veh_front, veh_adj_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "SLC", LC_related

        elif not veh_adj_front and veh_adj_rear:
            # OL prob
            OL_LC_prob, OL_LC_related = self._get_One_lead_LC_prob(veh_front, full_obs)
            
            # CI prob
            CI_LC_prob, CI_LC_related = self._get_Cut_in_LC_prob(veh_front, veh_adj_rear, full_obs)
            LC_related = CI_LC_related

            r_adj = self.position[0] - veh_adj_rear.position[0] - self.LENGTH

            if r_adj >= global_val.min_r_ignore:
                E_LC_prob = global_val.ignore_adj_veh_prob * OL_LC_prob + (1-global_val.ignore_adj_veh_prob) * CI_LC_prob
            else:
                E_LC_prob = CI_LC_prob
            return E_LC_prob, "Cut_in", LC_related

        elif veh_adj_front and veh_adj_rear:
            # SLC prob
            SLC_LC_prob, SLC_LC_related = self._get_Single_LC_prob(veh_front, veh_adj_front, full_obs)

            # DLC prob
            DLC_LC_prob, DLC_LC_related = self._get_Double_LC_prob(veh_adj_front, veh_adj_rear, full_obs)
            LC_related = DLC_LC_related

            r_adj = self.position[0] - veh_adj_rear.position[0] - self.LENGTH

            if r_adj >= global_val.min_r_ignore:
                E_LC_prob = global_val.ignore_adj_veh_prob * SLC_LC_prob + (1-global_val.ignore_adj_veh_prob) * DLC_LC_prob
            else:
                E_LC_prob = DLC_LC_prob
            return E_LC_prob, "DLC", LC_related


    # 20200421 Add Xintao This function do the NDD action for the new NDD data, therefore, some parameters and settings are not the same as previous
    # For example, now the velocity is the ego-vehicle, and the 50 meters determination for the SLC,DLC / CI,OL
    def Lateral_NDD_New(self, obs, modify_flag=True):
        """
        Decide the Lateral movement
        Input: observation of surrounding vehicles
        Output: whether do lane change (True, False), lane_change_idx (0:Left, 1:Still, 2:Right), action_pdf
        """
        initial_pdf = np.array([0,1,0])  # Left, Still, Right
        if not list(global_val.OL_pdf):
            # self._process_lane_change_data()
            assert("No OL_pdf file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs
        
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.one_lead_speed_list, round_item="speed")

        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehcile ahead
            left_prob, still_prob, right_prob = 0, 0, 0
            LC_related_list = []
            LC_type_list = []
            for item in ["Left", "Right"]:
                if item == "Left":
                    surrounding = (f1, f0, r0)
                    left_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
                else:
                    surrounding = (f1, f2, r2)
                    right_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
                
            # In the leftest or rightest
            if lane_id == 0:
                left_prob = 0
            if lane_id == 2:
                right_prob = 0
            if left_prob + right_prob > 1:
                tmp = left_prob + right_prob        
                left_prob *= 0.9/(tmp)
                right_prob *= 0.9/(tmp)                
            still_prob = 1 - left_prob - right_prob
            pdf_array = np.array([left_prob, still_prob, right_prob])
            if global_val.safety_guard_enabled_flag:          
                pdf_array = self._check_lateral_safety(obs, pdf_array)                  
            
            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)    
            to_lane_id = lane_id + lane_change_idx - 1
            if lane_change_idx != 1:
                if modify_flag:
                    self.target_lane_index = ("a", "b", to_lane_id)
                    if lane_change_idx == 0:
                        self.mode = LC_type_list[0]
                        self.LC_related = LC_related_list[0]
                        # print("LLC", self.mode, self.LC_related, pdf_array)
                    elif lane_change_idx == 2:
                        self.mode = LC_type_list[1]
                        self.LC_related = LC_related_list[1]
                        # print("RLC", self.mode, self.LC_related, pdf_array)
                return True, lane_change_idx, pdf_array
            else:
                return False, lane_change_idx, pdf_array