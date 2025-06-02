import gymnasium as gym

from .h1_matterport_base_cfg import H1MatterportBaseCfg, H1RoughPPORunnerCfg
from .h1_matterport_vision_cfg import H1MatterportVisionCfg, H1VisionRoughPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="h1_matterport_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1MatterportBaseCfg,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_matterport_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1MatterportVisionCfg,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)