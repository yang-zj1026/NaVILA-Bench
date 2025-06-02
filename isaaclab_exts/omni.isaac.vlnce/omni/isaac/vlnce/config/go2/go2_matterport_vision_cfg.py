import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns

import omni.isaac.vlnce.vlnce.mdp as mdp

from .go2_matterport_base_cfg import Go2MatterportBaseCfg, TerrainSceneCfg, Go2RoughPPORunnerCfg


@configclass
class Go2VisionRoughPPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_vision"


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        height_map = ObsTerm(
            func=mdp.height_map_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor"), "offset": 0.0},
            clip=(-10.0, 10.0),
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    
    @configclass
    class ProprioCfg(ObsGroup):
        """Observations for proprioceptive group."""

        # observation terms
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class CriticObsCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class CameraObsCfg(ObsGroup):
        """Observations for camera group."""

        # observation terms (order preserved)
        # depth_measurement = ObsTerm(
        #     func=mdp.process_depth_image,
        #     params={"sensor_cfg": SceneEntityCfg("lidar_sensor"), "data_type": "distance_to_image_plane"},
        # )
        rgb_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("rgbd_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class VizCameraObsCfg(ObsGroup):
        """Observations for visualization camera group."""
        rgb_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("viz_rgb_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    @configclass
    class DepthObsCfg(ObsGroup):
        """Observations for visualization camera group."""
        depth_measurement = ObsTerm(
            func=mdp.process_depth_image,
            params={"sensor_cfg": SceneEntityCfg("rgbd_camera"), "data_type": "distance_to_image_plane"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: CriticObsCfg = CriticObsCfg()
    camera_obs: CameraObsCfg = CameraObsCfg()
    viz_camera_obs: VizCameraObsCfg = VizCameraObsCfg()
    depth_obs: DepthObsCfg = DepthObsCfg()


##
# Scene configuration
##
class Go2VisionSceneCfg(TerrainSceneCfg):
    # lidar_sensor = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Head_lower",
    #     # offset=RayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.046), rot=(0., -0.991,0.0,-0.131)),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.0), rot=(0., -0.991,0.0,-0.131)),
    #     attach_yaw_only=False,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=32, vertical_fov_range=(0.0, 90.0), horizontal_fov_range=(-180, 180.0), horizontal_res=4.0
    #     ),
    #     debug_vis=False, # set to True to visualize the lidar rays
    #     mesh_prim_paths=["/World/matterport"],
    # )
    lidar_sensor = None

##
# Environment configuration
##

@configclass
class Go2MatterportVisionCfg(Go2MatterportBaseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    scene: Go2VisionSceneCfg = Go2VisionSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        # self.scene.lidar_sensor.update_period = 4*self.sim.dt
        self.scene.height_scanner.pattern_cfg.size = [3.0, 2.0]
