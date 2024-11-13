#!/bin/bash
#SBATCH --partition=batch                       
#SBATCH --mem=200G                             # Total memory for the job
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00                      

# Set ulimit to change the soft limit for virtual memory and data segment size
ulimit -v unlimited  # Remove the soft limit on virtual memory
ulimit -d unlimited  # Remove the soft limit on data segment size
ulimit -s unlimited  # Remove the limit on stack size

# Source the environment modules setup script
source /etc/profile.d/modules.sh || { echo "Failed to source environment modules setup script"; exit 1; }

# Load the Anaconda module and activate Conda environment
module load anaconda3 || { echo "Failed to load Anaconda module"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate IFAslurmv2 || { echo "Failed to activate Conda environment"; exit 1; }

# Run the Python script with arguments
# $1 represents the first argument (outputfolder) passed to the script
# $2 represents the second argument (fold) passed to the script
python /project/3022057.01/run_IFA/run_fold.py "$1" "$2"