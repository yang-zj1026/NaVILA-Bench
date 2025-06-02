# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import torch
import torch.nn as nn
import torchvision
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

# class DepthImageProcessor(nn.Module):
#     def __init__(self, image_height, image_width, num_output_units):
#         super(DepthImageProcessor, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * (image_height//4) * (image_width//4), num_output_units)  # Adjust based on your specific task
#         )
    
#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
# -- Navigation Action
class NavigationAction(ActionTerm):
    """Actions to navigate a robot by following some path."""

    cfg: NavigationActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: NavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # self.depth_cnn = DepthImageProcessor(image_height=self.cfg.image_size[0], 
        #                                      image_width=self.cfg.image_size[1],num_output_units=32).to(self.device)
        # self.resize_transform = torchvision.transforms.Resize((58, 87), 
        #                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        self.image_count = 0
        # # check if policy file exists
        if not check_file_path(self.cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{self.cfg.low_level_policy_file}' does not exist.")
        file_bytes = read_file(self.cfg.low_level_policy_file)

        # # # load policies
        self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.low_level_policy = torch.jit.freeze(self.low_level_policy.eval())

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = (
            3
        )  # [vx, vy, omega] --> vx: [-0.5,1.0], vy: [-0.5,0.5], omega: [-1.0,1.0]
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process high-level navigation actions. This function is called with a frequency of 10Hz"""

        # Store low level navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        # reshape into 3D path
        self._processed_navigation_velocity_actions[:] = actions.clone().view(self.num_envs, 3)

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
        #     # -- update command
            self._env.command_manager.compute(dt=self._low_level_step_dt)
            # Get low level actions from low level policy
            self._low_level_actions[:] = self.low_level_policy(
                self._env.observation_manager.compute_group(group_name="low_level_policy")
            )
            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class NavigationActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationAction
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""
    path_length: int = 51
    """Length of the path to be followed."""
    # low_level_agent_cfg: dict = {}
    image_size: tuple = ()
