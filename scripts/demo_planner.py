import argparse
import os
import cv2
import time
import math
import gzip, json
import numpy as np

# omni-isaaclab
from isaaclab.app import AppLauncher

import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")
parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")

parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# parser.add_argument("--draw_pointcloud", action="store_true", default=False, help="DRaw pointlcoud.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
# import ipdb; ipdb.set_trace()
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
import torch
from isaacsim.core.api.objects.cuboid import VisualCuboid

import gymnasium as gym
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper


def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)

    return roll, pitch, yaw


class Planner:
    def __init__(self, env, env_cfg, args_cli, simulation_app):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.simulation_app = simulation_app

        self.robot_start_pos = None
        self.vel_cmd_start_time = 0

        self.vel_command = torch.tensor([0.0, 0.0, 0.0], device = self.env.unwrapped.device)
        self.marker_cfg = CUBOID_MARKER_CFG.copy()
        self.marker_cfg.prim_path = "/Visuals/Command/pos_goal_command"
        self.marker_cfg.markers["cuboid"].scale = (0.5, 0.5, 0.5)
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.unwrapped.device).repeat(1, 1)
        self.data_npy = []

        for i in range(self.env_cfg.expert_path_length):
            expert_point_visualizer = VisualizationMarkers(self.marker_cfg)
            expert_point_visualizer.set_visibility(True)

            point = np.array(self.env_cfg.expert_path[i]).reshape(1, 3)
            default_scale = expert_point_visualizer.cfg.markers["cuboid"].scale
            # import ipdb; ipdb.set_trace()
            larger_scale = 2.0*torch.tensor(default_scale, device=self.env.unwrapped.device).repeat(1, 1)
            expert_point_visualizer.visualize(point, self.identity_quat,larger_scale)

    def pid_control(self, current_position, target_position, error_sum, prev_error, dt=0.2, Kp=0.5, Ki=0.00, Kd=0.000):
        """PID controller to compute velocity correction."""
        # target_position = np.array([target_position[0]])
        error = target_position[0] - current_position[0]
        # print("error: ", error)
        error_sum += error * dt
        error_diff = (error - prev_error) / dt
        correction = Kp * error + Ki * error_sum + Kd * error_diff
        prev_error = error
        return correction, error_sum, prev_error
    
    def compute_target_yaw(self, current_position, next_position):
        """Compute the target yaw angle based on the current and next position."""
        delta_position = next_position - current_position
        target_yaw = np.arctan2(delta_position[1], delta_position[0])
        return (target_yaw)%(2*np.pi)

    def start_loop(self):
        """Start the simulation loop."""

        # Set the camera view
        robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        roll, pitch, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        self.env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        # Reset the environment and apply zero velocity command
        obs, infos = self.env.reset()

        # Simulate physics
        start_it = 1
        it = 1

        start_t = time.time()
        self.start_time = time.time()
        self.robot_start_pos = robot_pos_w[:2]
        robot_yaw_quat = math_utils.yaw_quat(self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu()).unsqueeze(0)
        self.init_yaw_angle = math_utils.euler_xyz_from_quat(robot_yaw_quat)[2].numpy()

        sim_t = 0.0
        self.expert_vel_idx = 0
        planner_dt = self.env_cfg.decimation * self.env_cfg.sim.dt # 0.2
        self.traj_dt = 1.0
        # self.expert_velocity = np.diff(env_cfg.expert_path, axis=0)/self.traj_dt
        error_sum = 0.0
        prev_error = 0.0
        # num_frame = 0

        reached_goal_time = 0

        while self.simulation_app.is_running(): # 20hz
            # Save depth image
            rgb_image = infos['observations']['camera_obs'][0,:,:,:3].clone().detach()
            # save_path_rgb = os.path.join(os.getcwd(), "rgb_image"+str(it-start_it)+".png")
            rgb_image_np = rgb_image.cpu().numpy()
            rgb_image_np = cv2.rotate(rgb_image_np, cv2.ROTATE_90_CLOCKWISE)
            proprio_go2 = infos['observations']['policy'].clone().detach().cpu().numpy()

            robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            
            self.robot_pos = robot_pos_w[:2] - self.robot_start_pos
            robot_ori_full_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().unsqueeze(0)
            robot_ori_full_rpy = math_utils.euler_xyz_from_quat(robot_ori_full_quat)
            robot_yaw_quat = math_utils.yaw_quat(self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu()).unsqueeze(0)

            for i_ori in range(2):
                if robot_ori_full_rpy[i_ori][0] > math.pi:
                    robot_ori_full_rpy[i_ori][0] -= 2*math.pi
            
            # print("robot orientation: ", robot_ori_full_rpy)
            if abs(robot_ori_full_rpy[0].numpy()) > 0.6 or abs(robot_ori_full_rpy[1].numpy()) > 0.6:
                print("Large orientation: ", robot_ori_full_rpy[0], " ", robot_ori_full_rpy[1])
                return
            
            self.robot_yaw_angle = math_utils.euler_xyz_from_quat(robot_yaw_quat)[2].numpy() - self.init_yaw_angle
            robot_yaw_w = math_utils.euler_xyz_from_quat(robot_yaw_quat)[2].numpy()

            # env_cfg.expert_path = np.array(episode["gt_locations"])
            expert_path_length = len(env_cfg.expert_path)
            env_cfg.expert_time = np.arange(expert_path_length)*1.0

            relative_pos = torch.tensor(env_cfg.expert_path[self.expert_vel_idx] - robot_pos_w, device = self.env.unwrapped.device).to(torch.float32)
            expert_pos_body_frame = math_utils.quat_rotate_inverse(math_utils.yaw_quat(self.env.unwrapped.scene['robot'].data.root_quat_w), 
                                                         relative_pos.unsqueeze(0)).squeeze(0)
            
            correction, error_sum, prev_error = self.pid_control(
                robot_pos_w*0.0, expert_pos_body_frame, error_sum, prev_error
            )
            # command_velocity = actual_velocity + correction
            # import ipdb; ipdb.set_trace()
            self.vel_command[0] = torch.tensor(correction, device=self.env.unwrapped.device)
            self.vel_command[0] = torch.clamp(self.vel_command[0], min=0.0, max=0.5)

            # target_yaw = self.compute_target_yaw(robot_pos_w, env_cfg.expert_path[min(self.expert_vel_idx+1, env_cfg.expert_path_length-1)])
            target_yaw = self.compute_target_yaw(env_cfg.expert_path[self.expert_vel_idx], env_cfg.expert_path[min(self.expert_vel_idx+1, env_cfg.expert_path_length-1)])
            yaw_diff = target_yaw - robot_yaw_w
            if yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            elif yaw_diff >= np.pi:
                yaw_diff -= 2 * np.pi
            target_yaw_rate = np.clip(0.1 * yaw_diff / planner_dt, -0.5, 0.5)
            # print("target_yaw_rate: ", target_yaw_rate, " target_yaw: ", target_yaw, " robot_yaw_w: ", robot_yaw_w)
            # print("target vel x: ", self.vlm_vel_command[0], " robot_pos_w: ", robot_pos_w, " expert_pos: ", env_cfg.expert_path[self.expert_vel_idx])
            self.vel_command[2] = torch.tensor(target_yaw_rate, device=self.env.unwrapped.device)

            # obs[0, 6:9] = self.vel_command
            print("vel_command: ", self.vel_command)
            # self.env.update_command(self.vel_command)
            # import pdb; pdb.set_trace()
            # action = self.policy(obs)
            # proprioceptive_state = torch.tensor([robot_pos_w[0], robot_pos_w[1], robot_yaw_w, robot_gravity[0], robot_gravity[1], robot_gravity[2]], device=self.env.device)
            
            self.env.set_stop_called(reached_goal_time >= 10)
            obs, _, done, infos = self.env.step(self.vel_command)

            if done:
                print("Episode done!!!")
                break
            
            elapsed_time = time.time() - start_t
            # print("Time taken: ", elapsed_time)
            start_t = time.time()
            it += 1
            sim_t += planner_dt
            self.expert_vel_idx = min(int(sim_t / self.traj_dt), env_cfg.expert_path_length - 1)

            if np.linalg.norm(robot_pos_w[:2] - env_cfg.expert_path[-1][:2]) < 0.5:
                reached_goal = True
                reached_goal_time += 1
            else:
                reached_goal_time = 0

            if reached_goal_time > 10:
                print("Reached goal!!!")
                break
        
        # Print measurements
        print("\n============================== Episode Measurements ==============================")
        for key, value in infos["measurements"].items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    vel_command = torch.tensor([0, 0, 0])

    episode_idx = args_cli.episode_index
    dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    # scene_id = None
    with gzip.open(dataset_file_name, "rt") as f:
        deserialized = json.loads(f.read())
        episode = deserialized["episodes"][episode_idx]
        if "go2" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.4)
        elif "h1" in args_cli.task:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+1.0)
        else:
            env_cfg.scene.robot.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+0.5)

        env_cfg.scene.disk_1.init_state.pos = (episode["start_position"][0], episode["start_position"][1], episode["start_position"][2]+2.5)
        env_cfg.scene.disk_2.init_state.pos = (episode["reference_path"][-1][0], episode["reference_path"][-1][1], episode["reference_path"][-1][2]+2.5)
        wxyz_rot = episode["start_rotation"]
        init_rot = wxyz_rot
        # habitat2isaacsim_rot=math_utils.quat_from_euler_xyz(torch.tensor(0),torch.tensor(0),torch.tensor(0))
        # wxyz_rot = torch.tensor([episode["start_rotation"][3], episode["start_rotation"][0], episode["start_rotation"][1], episode["start_rotation"][2]])
        # convert from quaternion to euler angles
        # init_rot = math_utils.euler_xyz_from_quat(wxyz_rot)
        # euler_y_rotation = math_utils.euler_xyz_from_quat(wxyz_rot.unsqueeze(0))[1]
        # init_rot = math_utils.quat_from_euler_xyz(torch.tensor(0),torch.tensor(0),torch.tensor(euler_y_rotation))[0,:]
        # init_rot = episode["start_rotation"]
        # init_rot = [1.0,0.0,0.0,0.0]
        # init_rot=(init_rot[3], init_rot[0], init_rot[1], init_rot[2])
        env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
        # import ipdb; ipdb.set_trace()
        env_cfg.goals = episode["goals"]
        env_cfg.episode_id = episode["episode_id"]
        env_cfg.scene_id = episode["scene_id"].split('/')[1]
        env_cfg.traj_id = episode["trajectory_id"]
        env_cfg.instruction_text = episode["instruction"]["instruction_text"]
        env_cfg.instruction_tokens = episode["instruction"]["instruction_tokens"]
        env_cfg.reference_path = np.array(episode["reference_path"])
        expert_locations = np.array(episode["gt_locations"])
        # expert_locations=expert_locations[:,[0,2,1]]
        # expert_locations[:,1] = -expert_locations[:,1]
        # import ipdb; ipdb.set_trace()
        env_cfg.expert_path = expert_locations
        env_cfg.expert_path_length = len(env_cfg.expert_path)
        env_cfg.expert_time = np.arange(env_cfg.expert_path_length)*1.0
    # scene_id = "1LXtFkjw3qL"

    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")  

    print("scene_id: ", env_cfg.scene_id)
    print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
    
    # initialize environment and low-level policy
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # import pdb; pdb.set_trace()
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)  # Adjust device as needed
    ppo_runner.load(resume_path)

    low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    all_measures = ["PathLength", "DistanceToGoal", "Success", "SPL", "OracleNavigationError", "OracleSuccess"]
    env = VLNEnvWrapper(env, low_level_policy, args_cli.task, episode, high_level_obs_key="camera_obs",
                        measure_names=all_measures)

    planner = Planner(env, env_cfg, args_cli, simulation_app)
    planner.start_loop()
    # Close the simulator
    simulation_app.close()
    print("closed!!!")