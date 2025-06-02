import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils.configclass import configclass
from typing_extensions import Literal

from .goal_command_generator import GoalCommandGenerator


@configclass
class GoalCommandGeneratorCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = GoalCommandGenerator

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
