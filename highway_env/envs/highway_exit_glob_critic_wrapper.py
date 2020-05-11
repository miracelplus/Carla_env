from highway_env.envs.highway_exit_wrapper import HIGHWAYEXITCASE
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
import numpy as np
from gym.spaces import Box, Dict, Discrete, Tuple
import argparse
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

class HIGHWAYEXITCASE_Glob_Critic(HIGHWAYEXITCASE):
    
    def __init__(self, config):
        super(HIGHWAYEXITCASE_Glob_Critic, self).__init__(config)
        self.env = HIGHWAYEXITCASE(config)
        self.agent_ids = list(range(config["controlled_bv_num"]))
        observation_length = 1+1+config["controlled_bv_num"]+config["bv_observation_num"]
        obs_lower_bound_list = [config["min_distance"], config["min_lane"], config["min_velocity"]]*observation_length
        obs_upper_bound_list = [config["max_distance"], config["max_lane"], config["max_velocity"]]*observation_length
        self.observation_space = Dict({
            "own_obs": Box(np.array(obs_lower_bound_list), np.array(obs_upper_bound_list)),
            "opponent_obs": Box(np.array(obs_lower_bound_list*4), np.array(obs_upper_bound_list*4)),
            "opponent_action": Tuple([Discrete(11)]*4),
        })

    def reset(self):
        obs_dict = self.env.reset()
        return self.to_global_obs(obs_dict)

    def step(self, action_dict):
        obs_dict, rewards, dones, infos = self.env.step(action_dict)
        return self.to_global_obs(obs_dict), rewards, dones, infos

    def to_global_obs(self, obs_dict):
        obs0 = obs_dict.copy()
        obs1 = obs_dict.copy()
        obs2 = obs_dict.copy()
        obs3 = obs_dict.copy()
        obs4 = obs_dict.copy()
        del obs0["car0"]
        del obs1["car1"] 
        del obs2["car2"]
        del obs3["car3"]
        del obs4["car4"]
        obs0_array, obs1_array, obs2_array, obs3_array, obs4_array = [], [], [], [], []
        for key in obs0:
            obs0_array.append(obs0[key])
        for key in obs1:
            obs1_array.append(obs1[key])
        for key in obs2:
            obs2_array.append(obs2[key])
        for key in obs3:
            obs3_array.append(obs3[key])
        for key in obs4:
            obs4_array.append(obs4[key])
        obs0 = np.array(obs0_array).flatten()
        obs1 = np.array(obs1_array).flatten()
        obs2 = np.array(obs2_array).flatten()
        obs3 = np.array(obs3_array).flatten()
        obs4 = np.array(obs4_array).flatten()

        return {
            "car0":{
                "own_obs": obs_dict["car0"], 
                "opponent_obs": obs0, 
                "opponent_action": [0,0,0,0],
                },
            "car1":{
                "own_obs": obs_dict["car1"], 
                "opponent_obs": obs1, 
                "opponent_action": [0,0,0,0],
                },
            "car2":{
                "own_obs": obs_dict["car2"], 
                "opponent_obs": obs2, 
                "opponent_action": [0,0,0,0],
                },
            "car3":{
                "own_obs": obs_dict["car3"], 
                "opponent_obs": obs3, 
                "opponent_action": [0,0,0,0],
                },
            "car4":{
                "own_obs": obs_dict["car4"], 
                "opponent_obs": obs4, 
                "opponent_action": [0,0,0,0],
                }
        }


class CentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized VF.
    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        observation_length = 12
        obs_lower_bound_list = [0, 0, 20]*observation_length
        obs_upper_bound_list = [650, 2, 40]*observation_length
        self.action_model = FullyConnectedNetwork(
            Box(np.array(obs_lower_bound_list), np.array(obs_upper_bound_list)),  # one-hot encoded Discrete(6)
            action_space,
            num_outputs,
            model_config,
            name + "_action")
        self.register_variables(self.action_model.variables())

        self.value_model = FullyConnectedNetwork(obs_space, action_space, 1,
                                                 model_config, name + "_vf")
        self.register_variables(self.value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model({
            "obs": input_dict["obs_flat"]
        }, state, seq_lens)
        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
        }, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])





