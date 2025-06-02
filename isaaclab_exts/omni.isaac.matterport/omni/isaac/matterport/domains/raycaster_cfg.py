# Copyright (c) 2024-2025 VLN-CE Project Developers
# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from .matterport_raycaster import MatterportRayCaster


@configclass
class MatterportRayCasterCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor for Matterport Environments."""

    class_type = MatterportRayCaster
    """Name of the specific matterport ray caster class."""
