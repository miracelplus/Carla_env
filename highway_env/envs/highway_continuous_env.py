from highway_env.envs.highway_exit_BV_env import *
from highway_env.vehicle.behavior import IDM_Continuous_Vehicle
class HighwayExitEnvNDD_Continuous(HighwayExitEnvNDD_DBV):
    def __init__(self,config):
        super(HighwayExitEnvNDD_Continuous,self).__init__(config)
    
    def _make_vehicles(self, background_vehicle=None,
                       auto_vehicle=(0, global_val.initial_CAV_position, global_val.initial_CAV_speed)):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        background_vehicle: Each item has [lane,position,velocity]
        """
        ego_vehicle = MDPVehicle(self.road, self.road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0), velocity=auto_vehicle[2])
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        other_vehicles_type = IDM_Continuous_Vehicle

        if self.generate_vehicle_mode == "Random":
            for _ in range(self.config["vehicles_count"]):
                new_vehicle = self.create_random_acc_training(other_vehicles_type)
                if new_vehicle:
                    self.road.vehicles.append(new_vehicle)
    
    def _simulate(self, cav_action=None, bv_action=None):
        """
            Perform several steps of simulation with constant action
        """
        for k in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            if ((cav_action is not None) or (bv_action is not None)) and \
                    self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"]) == 0:
                # Set the CAV and BV action
                self.vehicle.act(self.ACTIONS[cav_action])
                for i in range(len(self.controlled_bvs)):
                    bv = self.controlled_bvs[i]
                    bv.act(bv_action[i])
            #  when nothing happens, vehicle act nothing
            self.vehicle.act()
            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1
            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()
            road_crash_flag = False
            for vehicle in self.road.vehicles:
                if vehicle.crashed:
                    road_crash_flag = True
                    break
            if road_crash_flag:
                break  
        self.enable_auto_render = False