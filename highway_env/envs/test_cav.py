import gym
from gym.spaces import Discrete, Box
from ray import tune
from highway_env.envs.highway_exit_ndd_env import HighwayExitEnvNDD
import numpy as np 
env_config = {"user":"ENV", "mode":"NDD","generate_vehicle_mode":"Random",\
        "min_distance":0,"max_distance":500,"min_velocity":20,"max_velocity":40,\
            "min_lane":0,"max_lane":2}}
env = HighwayExitEnvNDD(env_config)

test_item = 0
effective_item = 0
crash_num = 0
while(1):
    test_item += 1
    env.reset()
    action = np.random.randint(0,254)
    observation, reward, done, info = env.step(action)
    if done:
        if reward > 50:
            crash_num += 1
            effective_item += 1
        elif reward > -100:
            effective_item += 1
    if effective_item == 100:
        crash_rate = crash_num/effective_item
        break
print(crash_rate)



