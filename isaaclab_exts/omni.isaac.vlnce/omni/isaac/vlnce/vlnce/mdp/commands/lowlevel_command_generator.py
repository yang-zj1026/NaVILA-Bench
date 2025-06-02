# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""
from __future__ import annotations

import os
import math
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    CUBOID_MARKER_CFG,
)
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import check_file_path, read_file
# from isaaclab.utils.assets import LOCAL_ISAAC_DIR


if TYPE_CHECKING:
    from .lowlevel_command_generator_cfg import LowLevelCommandGeneratorCfg


class LowLevelCommandGenerator(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: LowLevelCommandGeneratorCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: LowLevelCommandGeneratorCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg (LowLevelCommandGeneratorCfg): The configuration of the command generator.
            env (object): The environment.
        """
        super().__init__(cfg, env)
        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        self.twist: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        # -- debug vis
        self._base_vel_goal_markers = None
        self._base_vel_markers = None

        # Rotation mark
        self.rotation_mark = False
        self.initialized = False
        self.goal_reached = False
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._env.device).repeat(self._env.num_envs, 1)

        self.objnav_policy_path = os.path.join(LOCAL_ISAAC_DIR, "low_level_policy", "mid_policy_height_scan.jit")
        self.objnav_policy_path = os.path.join(LOCAL_ISAAC_DIR, "low_level_policy", "mid_policy_depth.jit")
        if not check_file_path(self.objnav_policy_path):
            raise FileNotFoundError(f"Policy file '{self.objnav_policy_path}' does not exist.")
        file_bytes = read_file(self.objnav_policy_path)
        self.mid_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.mid_level_policy = torch.jit.freeze(self.mid_level_policy.eval())


    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "rlCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tLookahead distance: {self.cfg.lookAheadDistance}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        # print("twist: ", self.twist)
        return self.twist

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Reset the command generator.

        This function resets the command generator. It should be called whenever the environment is reset.

        Args:
            env_ids (Optional[Sequence[int]], optional): The list of environment IDs to reset. Defaults to None.
        """
        if env_ids is None:
            env_ids = ...

        self.twist = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached = False

        return {}

    def compute(self, dt: float):
        """Compute the command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. The paths
        should be given in the base frame of the robot. Num_envs is equal to the number of robots spawned in all
        environments.
        """
        if not self.rotation_mark and not self.goal_reached and self.initialized:
            # get paths
            self.twist = self.mid_level_policy(self._env.observation_manager.compute_group(group_name="mid_level_planner")).reshape(self._env.num_envs, 3)
            # self.twist = self._env.action_manager._terms['paths']._processed_navigation_velocity_actions.clone()
            # import ipdb; ipdb.set_trace()
            self.twist[:, 0] = torch.clip(self.twist[:, 0], 0.0, 0.5)
            self.twist[:,1] = 0.0
            self.twist[:,2] = torch.clip(self.twist[:,2], -math.pi, math.pi)
        elif self.goal_reached or (not self.initialized):
            # self.twist[:, 0] = torch.clip(self.twist[:, 0], 0.0, 0.5)
            # self.twist[:,1] = 0.0
            self.twist[:,:2] = torch.zeros((self.num_envs, 2), device=self.device)
        else:
            # self.twist = torch.zeros((self.num_envs, 3), device=self.device)
            self.twist[:,:2] = torch.zeros((self.num_envs, 2), device=self.device)
            self.twist[:, 2] += 0.1
            print("rotation")
        self.goal_reached = self._env.command_manager._terms['midlevel_command'].command[:, :2].norm(dim=1) < 2.5
        return self.twist

    """
    Implementation specific functions.
    """

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
                # -- goal command
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
                marker_cfg.markers["cuboid"].scale = (0.5, 0.5, 0.5)
                # self.base_vel_goal_command_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
            # self.base_vel_goal_command_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)
                # self.base_vel_goal_command_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        base_quat_w = self.robot.data.root_quat_w.clone()

        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        # pos_command_w = self._env.command_manager._terms['midlevel_command'].pos_command_w.clone()
        # default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # larger_scale = 2.0*torch.tensor(default_scale, device=self.device).repeat(pos_command_w.shape[0], 1)
        # self.base_vel_goal_command_visualizer.visualize(pos_command_w, self.identity_quat,larger_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1)
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat
