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
)
from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    from .path_follower_command_generator_cfg import PathFollowerCommandGeneratorCfg


class PathFollowerCommandGenerator(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: PathFollowerCommandGeneratorCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: PathFollowerCommandGeneratorCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg (PathFollowerCommandGeneratorCfg): The configuration of the command generator.
            env (object): The environment.
        """
        super().__init__(cfg, env)
        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        # -- buffers
        self.vehicleSpeed: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.switch_time: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.vehicleYawRate: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.navigation_forward: torch.Tensor = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.twist: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        # -- debug vis
        self._base_vel_goal_markers = None
        self._base_vel_markers = None

        # Rotation mark
        self.rotation_mark = False
        self.initialized = False
        self.goal_reached = False

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "PathFollowerCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tLookahead distance: {self.cfg.lookAheadDistance}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
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

        self.vehicleSpeed = torch.zeros(self.num_envs, device=self.device)
        self.switch_time = torch.zeros(self.num_envs, device=self.device)
        self.vehicleYawRate = torch.zeros(self.num_envs, device=self.device)
        self.navigation_forward = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.twist = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached = False

        return {}

    def compute(self, dt: float):
        """Compute the command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. The paths
        should be given in the base frame of the robot. Num_envs is equal to the number of robots spawned in all
        environments.
        """
        # get paths
        paths = self._env.action_manager._terms['paths']._processed_navigation_velocity_actions.clone()
        # get number of pases of the paths
        num_envs, N = paths.shape
        assert N > 0, "PathFollowerCommandGenerator: paths must have at least one poses."
        # get the current simulation time
        curr_time = self.sim.current_time
        # define current maxSpeed for the velocities
        max_speed = torch.ones(num_envs, device=self.device) * self.cfg.maxSpeed

        # # transform path in base/ robot frame if given in world frame
        # if self.cfg.path_frame == "world":
        #     paths = math_utils.quat_apply(
        #         math_utils.quat_inv(self.robot.data.root_quat_w[:, None, :].repeat(1, N, 1)),
        #         paths - self.robot.data.root_pos_w[:, None, :],
        #     )
        self.paths_diff_global = paths[:,:2] - self.robot.data.root_pos_w[:, :2]

        if self.initialized and not self.goal_reached:
            if not self.rotation_mark:
                # if abs(self.paths_diff_global[0,0,0]) < 0.5:
                #     self.twist[:, 0] = 0.0
                # else:
                self.twist[:, 0] = min(max(self.paths_diff_global[0,0], -0.5), 0.5)
                # if abs(self.paths_diff_global[0,0,1]) < 2.0:
                #     self.twist[:, 1] = 0.0
                # else:
                self.twist[:, 1] = min(max(self.paths_diff_global[0,1], -0.3), 0.3)
                self.twist[:, 2] = min(max(0.01*paths[0,2], -0.2), 0.2)
                # TODO: add yaw rotation mechanism
            else:
                self.twist[:,0] = 0.0
                self.twist[:,1] = 0.0
                self.twist[:,2] = 0.5
        else:
            self.twist[:,0] = 0.0
            self.twist[:,1] = 0.0
            self.twist[:,2] = 0.0
        if (torch.linalg.norm(self.paths_diff_global[0,:2], dim=-1))<3.0 and self.initialized:
            self.goal_reached = True
        # print("goal_reached: ", self.goal_reached, " rotation_mark: ", self.rotation_mark, " twist: ", self.twist[:,:3], " norm: ", torch.linalg.norm(self.paths_diff_global[0,0,:2], dim=-1), " initialized: ", self.initialized)

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
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

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
