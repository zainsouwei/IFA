import os
import sys
import argparse

sys.path.append('/project/3022057.01/IFA/utils')
from preprocessing import parcellate, times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HCP fMRI task")
    parser.add_argument("task", help="Task name (e.g., GAMBLING)")
    args = parser.parse_args()

    task = args.task
    output_dir = os.path.join("/project/3022057.01/HCP", task)
    os.makedirs(output_dir, exist_ok=True)
    print("You passed this task", task)
    print("This is the output directory", output_dir)

    base_directory = "/project_cephfs/3022017.01/S1200"

    # Run parcellation to process subjects and generate the metadata DataFrame
    full_data = parcellate(task, output_dir, base_directory=base_directory)
    
    # Compute the condition indices for the given task and update the DataFrame
    times(full_data, output_dir, task)

