from highway_env.envs.highway_exit_wrapper import *
from highway_env.envs.highway_continuous_env import *

class HIGHWAYEXITCASE_Continuous(HIGHWAYEXITCASE):
    def __init__(self, config):
        super(HIGHWAYEXITCASE_Continuous, self).__init__(config)
        self.base_env = HighwayExitEnvNDD_Continuous(config)
        self.CAV_agent = cagent.agent(self.base_env)
        self.action_space = Box(-8,8,shape=(1,), dtype=np.float32)
        self.agent_ids = list(range(config["controlled_bv_num"]))
        #for i in range(config["controlled_bv_num"]):
