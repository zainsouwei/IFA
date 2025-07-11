#!/bin/bash
#SBATCH --partition=batch                       
#SBATCH --mem=250G                             
#SBATCH --time=72:00:00
#SBATCH --ntasks 1     
#SBATCH --cpus-per-task=15                 

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
python /project/3022057.01/IFA/run_IFA/Partial/run_fold_partial.py "$1" "$2"