from highway_env.envs.highway_exit_BV_env import *


class HighwayExitEnvNDD_DBV_Dyn(HighwayExitEnvNDD_DBV):
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
        super(HighwayExitEnvNDD_DBV_Dyn, self).__init__(config)
        
    def step(self, action):
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        assert len(bv_action) == self.controlled_bv_num
        self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, exit_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag":cav_crash_flag, "bv_crash_flag":bv_crash_flag, "exit_flag":exit_flag, "bv_crash_index":bv_crash_index, "cav_action":cav_action}
        # choose new controlled bv
        self.determine_controlled_bv()
        # get reward and observation
        cav_reward = self._reward_cav(exit_flag, done)
        bv_reward, infos = self._reward_bv_multi(info, done)
        observation = self.observe_cav_bv()
        reward = Reward(cav_reward=cav_reward, bv_reward=bv_reward)
        for bv in self.road.vehicles[1:]:
            bv.actual_action = False
        return observation, reward, done, infos

