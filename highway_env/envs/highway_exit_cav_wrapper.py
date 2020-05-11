from highway_env.envs.highway_exit_cav_env import *
# This env is designed for ENV based on highway-exit-ndd-env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import CAV_agent.agent as cagent
import gym
from gym.spaces import Discrete, Box


class HIGHWAYEXITCASE_CAV():
    def __init__(self, config):
        self.config = config
        self.base_env_cav = HighwayExitEnvNDD_CAV(config)
        self.action_space = Discrete(5)
        # We have observation of CAV, controlled BVs, BV adjacent Vehicle 1+controlled_bv_num+1+bv_observation_num
        observation_length = 1+self.config["cav_observation_num"]
        obs_lower_bound_list = [self.config["min_distance"], self.config["min_lane"], self.config["min_velocity"]]*observation_length
        obs_upper_bound_list = [global_val.EXIT_LENGTH, self.config["max_lane"], self.config["max_velocity"]]*observation_length
        self.observation_space = Box(np.array(obs_lower_bound_list), np.array(obs_upper_bound_list))

    def reset(self):
        observation = self.base_env_cav.reset()
        return observation.cav_observation

    def step(self, action):
        cav_action = action
        bv_action = None
        new_action = Action(cav_action=cav_action,bv_action=bv_action)
        obs, reward, done, info = self.base_env_cav.step(new_action)
        reward = reward.cav_reward
        obs = obs.cav_observation
        return obs, reward, done, info

    def render(self):
        self.base_env_cav.render()


