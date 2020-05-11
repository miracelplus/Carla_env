from highway_env.envs.highway_exit_BV_env import *
# This env is designed for ENV based on highway-exit-ndd-env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import CAV_agent.agent as cagent
import gym
from gym.spaces import Discrete, Box
import copy


class HIGHWAYEXITCASE(MultiAgentEnv):
    def __init__(self, config):
        self.config = config
        self.base_env = HighwayExitEnvNDD_DBV(config)
        self.vehicle = self.base_env.vehicle
        self.CAV_agent = cagent.agent(self.base_env,CAV_eval_flag=True, relative_state=False)
        self.action_space = Discrete(33)
        self.bv_observation_num = config["bv_observation_num"]
        # We have observation of CAV, controlled BVs, BV adjacent Vehicle 1+controlled_bv_num+1+bv_observation_num
        observation_length = 1+1+self.config["bv_observation_num"]
        # observation_length = 2
        self.obs_lower_bound_list = [0, self.config["min_lane"], self.config["min_velocity"]]*observation_length
        self.obs_upper_bound_list = [global_val.EXIT_LENGTH - global_val.initial_CAV_position, self.config["max_lane"], self.config["max_velocity"]]*observation_length
        self.observation_space = Box(np.array(self.obs_lower_bound_list), np.array(self.obs_upper_bound_list))
    
    def save_base_env(self):
        return copy.deepcopy(self.base_env)
    
    def load_base_env(self, base_env):
        self.base_env = copy.deepcopy(base_env)
        
     
    def whole_observation_2_bv_observation_dict(self, whole_obs):
        bv_observation = whole_obs[0].bv_observation #nparray matrix
        bv_indicator = whole_obs[1].bv_indicator
        assert self.base_env.controlled_bv_num == len(bv_observation)
        obs = {}
        for i in range(self.base_env.controlled_bv_num):
            car_name_str = "car"+str(i)
            obs[car_name_str] = [bv_observation[i], bv_indicator[i]]
        return obs

    def reset(self, given_ini=None):
        observation, ini_data = self.base_env.reset(given_ini=given_ini)
        return self.whole_observation_2_bv_observation_dict(observation), ini_data
        
    def load_env(self):
        self.base_env.determine_controlled_bv()
        observation = self.base_env.observe_cav_bv()
        return self.whole_observation_2_bv_observation_dict(observation)

    def reward_list_2_reward_dict(self, reward_list):
        reward_dict = {}
        assert self.base_env.controlled_bv_num == len(reward_list)
        for i in range(len(reward_list)):
            car_item_num = "car"+str(i)
            reward_dict[car_item_num] = reward_list[i]
        return reward_dict



    def step(self, action):
        bv_action = []
        for i in range(self.base_env.controlled_bv_num):
            car_name_str = "car"+str(i)
            bv_action.append(action[car_name_str])
        cav_obs, action_indicator = self.base_env.get_cav_observation()
        _, cav_action = self.CAV_agent.decision(cav_obs, action_indicator=action_indicator)
        # cav_action = 6
        new_action = Action(cav_action=cav_action,bv_action=bv_action)
        observation, reward, done, infos, weight = self.base_env.step(new_action)
        rewards = self.reward_list_2_reward_dict(reward.bv_reward)
        new_obs_dict = self.whole_observation_2_bv_observation_dict(observation)
        dones = {}
        dones["__all__"] = done
        # for i in range(len(self.base_env.road.vehicles)):
        #     bv = self.base_env.road.vehicles[i]
        #     if bv.crashed:
        #         car_name = "car"+str(i)
        #         dones[car_name] = (str(bv.position[0]), str(bv.position[1]))
        for i in range(self.base_env.controlled_bv_num):
            bv = self.base_env.controlled_bvs[i]
            if bv.crashed:
                car_name = "car"+str(i)
                dones[car_name] = True
        infos["cav_obs"] = cav_obs
        return new_obs_dict, rewards, dones, infos, weight, cav_action
        # cav_new_obs = self.base_env.get_cav_observation()
        # return cav_new_obs, rewards, dones, infos, weight, cav_action

    def render(self):
        self.base_env.render()


