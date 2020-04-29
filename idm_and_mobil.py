# IDM model
# https://en.wikipedia.org/wiki/Intelligent_driver_model

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
    acceleration = self.COMFORT_ACC_MAX * (
            1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), self.DELTA))
    if front_vehicle:
        d = ego_vehicle.lane_distance_to(front_vehicle)
        acceleration -= self.COMFORT_ACC_MAX * \
            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
    return acceleration

# MOBIL model
# Can be seen in the pdf file


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
        jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                            + old_following_pred_a - old_following_a)
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

    # All clear, let's go!
    return True