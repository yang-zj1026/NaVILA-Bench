import gymnasium as gym

from .go2_matterport_base_cfg import Go2MatterportBaseCfg, Go2RoughPPORunnerCfg
from .go2_matterport_vision_cfg import Go2MatterportVisionCfg, Go2VisionRoughPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="go2_matterport_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2MatterportBaseCfg,
        "rsl_rl_cfg_entry_point": Go2RoughPPORunnerCfg,
    },
)

gym.register(
    id="go2_matterport_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2MatterportVisionCfg,
        "rsl_rl_cfg_entry_point": Go2VisionRoughPPORunnerCfg,
    },
)