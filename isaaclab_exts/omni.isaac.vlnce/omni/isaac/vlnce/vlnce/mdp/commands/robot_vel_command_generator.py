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
from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    from .robot_vel_command_generator_cfg import RobotVelCommandGeneratorCfg


class RobotVelCommandGenerator(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: RobotVelCommandGeneratorCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: RobotVelCommandGeneratorCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg (RLCommandGeneratorCfg): The configuration of the command generator.
            env (object): The environment.
        """
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        self.command_b: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        # -- debug vis
        self._base_vel_goal_markers = None
        self._base_vel_markers = None

        # Rotation mark
        self.rotation_mark = False
        self.initialized = False

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
        return self.command_b

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

        self.command_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached = False

        return {}

    def compute(self, dt: float):
        """Compute the command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. The paths
        should be given in the base frame of the robot. Num_envs is equal to the number of robots spawned in all
        environments.
        """
        # get paths
        self.command_b[:,:3] = self._env.action_manager._terms['vlm_actions']._processed_command_velocity_actions.clone()[:,:3]

        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass
