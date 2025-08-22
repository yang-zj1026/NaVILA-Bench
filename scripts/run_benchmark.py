import os
import argparse
import subprocess
import numpy as np
import gzip
import json


def read_episodes(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["episodes"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-data-path", type=str, default="isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz")
    parser.add_argument("--navila-model-path", type=str, default="/home/zhaojing/mnt/legged_nav/NaVILA/NaVILA-llama3-8B-8f-scanqa-rxr")
    parser.add_argument("--task", type=str, default="go2_matterport_vision")
    parser.add_argument("--low_level_policy_dir", type=str, default="2024-09-25_23-22-02")
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()
    
    # Define the arguments for evaluation
    eval_args = [f"--task={args.task}", "--num_envs=1",
                f"--load_run={args.low_level_policy_dir}", 
                "--headless", "--enable_cameras", 
                #  "--visualize_path"
                ]
    
    if args.task == "go2_matterport_vision":
        eval_args.append("--history_length=9")

    episodes = read_episodes(args.r2r_data_path)

    for i in range(args.start_idx, len(episodes)):
        episode = episodes[i]
        scene_id = episode['scene_id']
        print("Episode id: ", episode['episode_id'])

        msg = f"\n======================= Running Evaluation of Episode {i} ======================="
        msg += f"\nScene: {episodes[i]['scene_id']}"
        msg += f"\nStart Position: {episodes[i]['start_position']}"
        msg += f"\nStart Rotation: {episodes[i]['start_rotation']}"
        msg += f"\nInstruction: {episodes[i]['instruction']['instruction_text']}\n"
        print(msg)
        
        eval_args.append(f"--episode_idx={i}")
        subprocess.run(['python', 'scripts/navila_eval.py'] + eval_args)