from highway_env.envs.highway_exit_bv_dynamic import *
from highway_env.envs.highway_exit_wrapper import *

class HIGHWAYEXITCASE_Dyn(HIGHWAYEXITCASE):
    def __init__(self, config):
        super(HIGHWAYEXITCASE_Dyn, self).__init__(config)
        self.base_env = HighwayExitEnvNDD_DBV_Dyn(config)
        self.CAV_agent = cagent.agent(self.base_env)