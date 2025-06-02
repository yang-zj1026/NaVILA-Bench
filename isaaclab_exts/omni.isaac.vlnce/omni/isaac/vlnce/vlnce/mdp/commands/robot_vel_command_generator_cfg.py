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

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass
from typing_extensions import Literal

from .robot_vel_command_generator import RobotVelCommandGenerator


@configclass
class RobotVelCommandGeneratorCfg(CommandTermCfg):
    class_type: RobotVelCommandGenerator = RobotVelCommandGenerator
    """Name of the command generator class."""

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""

    # path_frame: Literal["world", "robot"] = "world"
    """Frame in which the path is defined.
    - ``world``: the path is defined in the world frame. Also called ``odom``.
    - ``robot``: the path is defined in the robot frame. Also called ``base``.
    """

