#!/bin/bash
#SBATCH --partition=batch                       
#SBATCH --mem=250G                             
#SBATCH --time=8:00:00                        
#SBATCH --output=/project/3022057.01/slurm-%j.out  
#SBATCH --error=/project/3022057.01/slurm-%j.err   

# Remove limits on virtual memory, data segment, and stack sizes
ulimit -v unlimited  
ulimit -d unlimited  
ulimit -s unlimited  

# Source the environment modules setup script
source /etc/profile.d/modules.sh || { echo "Failed to source environment modules setup script"; exit 1; }

# Load the Anaconda module and activate the Conda environment
module load anaconda3 || { echo "Failed to load Anaconda module"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate IFAslurmv2 || { echo "Failed to activate Conda environment"; exit 1; }

# Change to the project directory
cd /project/3022057.01/ || { echo "Failed to change directory"; exit 1; }

# Check if a task argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <TASK>"
    exit 1
fi

TASK="$1"

# Pass the task argument to the Python script
python /project/3022057.01/IFA/utils/parcellate.py "$TASK"
