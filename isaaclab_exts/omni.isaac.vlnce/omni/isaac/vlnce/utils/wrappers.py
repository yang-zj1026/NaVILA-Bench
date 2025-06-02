# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch
import numpy as np

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .measures import add_measurement


def get_proprio_obs_dim(env: ManagerBasedRLEnv) -> int:
    """Returns the dimension of the proprioceptive observations."""
    return env.unwrapped.observation_manager.compute_group("proprio").shape[1]


class RslRlVecEnvHistoryWrapper(RslRlVecEnvWrapper):
    """Wraps around Isaac Lab environment for RSL-RL to add history buffer to the proprioception observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv, history_length: int = 1):
        """Initializes the wrapper."""
        super().__init__(env)

        self.history_length = history_length
        self.proprio_obs_dim = get_proprio_obs_dim(env)
        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.proprio_obs_dim,
                                                    dtype=torch.float, device=self.unwrapped.device)
        
        self.clip_actions = 20.0

    """
    Properties
    """
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        self.proprio_obs_buf = torch.cat([proprio_obs.unsqueeze(1)] * self.history_length, dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        obs_dict["policy"] = curr_obs

        return curr_obs, {"observations": obs_dict}
    
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Resets the environment."""
        obs_dict, infos = self.env.reset()
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        self.proprio_obs_buf = torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1)
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        infos["observations"] = obs_dict

        return curr_obs, infos

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip the actions (for testing only)
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        # print("============== Height Map ==============")
        # print(obs_dict["test_height_map"])
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # update obsservation history buffer & reset the history buffer for done environments
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None], 
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                self.proprio_obs_buf[:, 1:],
                proprio_obs.unsqueeze(1)
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        # return the step information
        return curr_obs, rew, dones, extras

    def update_command(self, command: torch.Tensor) -> None:
        """Updates the command for the environment."""
        self.proprio_obs_buf[:, -1, 6:9] = command

    def close(self):  # noqa: D102
        return self.env.close()


class VLNEnvWrapper:
    """Wrapper to configure an :class:`ManagerBasedRLEnv` instance to VLN environment."""

    def __init__(self, env: ManagerBasedRLEnv, 
                 low_level_policy, task_name, 
                 episode, max_length=10000, high_level_obs_key="camera_obs",
                 measure_names=["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
        ):
        self.env = env
        self.task_name = task_name
        self.episode = episode
        self.measure_names = measure_names

        self.env_step = 0
        self.max_length = max_length

        self.high_level_obs_key = high_level_obs_key
        assert high_level_obs_key in self.env.observation_space.spaces.keys() # CHECK this

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.curr_pos, self.prev_pos = None, None
        self.is_stop_called = False

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def set_measures(self):
        self.measure_manager = add_measurement(self.env, self.episode, self.measure_names)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        low_level_obs, infos = self.env.reset()
        self.low_level_obs = low_level_obs
        zero_cmd = torch.tensor([0., 0., 0.], device=low_level_obs.device)

        if "go2" in self.task_name:
            warmup_steps = 100
        elif "h1" or "g1" in self.task_name:
            warmup_steps = 200
        else:
            warmup_steps = 50

        for i in range(warmup_steps):
            if i % 100 == 0 or i == warmup_steps - 1:
                print(f"Warmup step {i}/{warmup_steps}...")

            self.update_command(zero_cmd)
            actions = self.low_level_policy(self.low_level_obs)
            low_level_obs, _, _, infos = self.env.step(actions)
            self.low_level_obs = low_level_obs
            self.low_level_action = actions

        self.env_step, self.same_pos_count = 0, 0
        
        self.set_measures()
        self.measure_manager.reset_measures()
        measurements = self.measure_manager.get_measurements()
        infos["measurements"] = measurements

        self.prev_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()

        obs = infos["observations"][self.high_level_obs_key]
        return obs, infos
    
    def update_command(self, command) -> None:
        """Update the command for the low-level policy."""

        # make sure command is a tensor on the same device as low_level_obs
        if not torch.is_tensor(command):
            command = torch.tensor(command, device=self.env.unwrapped.device)

        if isinstance(self.env, RslRlVecEnvHistoryWrapper):
            self.low_level_obs[:, 6:9] = command
            self.env.proprio_obs_buf[:, -1, 6:9] = command
        
        else:
            self.low_level_obs[:, 9:12] = command

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take a step in the environment.

        Args:
            action: The action of high-level planner, which should be velocity command to the low-level policy.

        Returns:
            obs: The observation of the high-level planner.
            reward: The reward of the environment.
            done: Whether the episode is done.
            info: Additional information of the environment.
        
        """

        self.update_command(action)

        low_level_action = self.low_level_policy(self.low_level_obs)
        self.low_level_action = low_level_action

        low_level_obs, reward, done, info = self.env.step(low_level_action)
        self.low_level_obs = low_level_obs
        obs = info["observations"][self.high_level_obs_key]
        self.env_step += 1

        self.measure_manager.update_measures()
        measurements = self.measure_manager.get_measurements()
        info["measurements"] = measurements

        # Check if the robot has stayed in the same location for 1000 steps or env has reached max length
        same_pos = self.check_same_pos()
        done = done[0] or same_pos or self.env_step >= self.max_length

        return obs, reward, done, info
    
    def check_same_pos(self) -> bool:
        curr_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        robot_vel = torch.norm(self.env.unwrapped.scene["robot"].data.root_vel_w[0].detach())
        if torch.norm(curr_pos - self.prev_pos) < 0.01 and robot_vel < 0.1:
            self.same_pos_count += 1
        else:
            self.same_pos_count = 0
        self.prev_pos = curr_pos

        # Break out of the loop if the robot has stayed in the same location for 1000 steps
        if self.same_pos_count >= 1000:
            print("Robot has stayed in the same location for 1000 steps. Breaking out of the loop.")
            return True
        
        return False

    def set_stop_called(self, is_stop_called: bool) -> None:
        """Set the stop called flag."""
        self.env.is_stop_called = is_stop_called
    
    def close(self) -> None:
        self.env.close()

    