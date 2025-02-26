#!/bin/bash
#SBATCH --partition=gpu                       # Use the GPU partition
#SBATCH --gpus=nvidia_a100_80gb_pcie:1        # Request 1 NVIDIA A100 GPU 80GB
#SBATCH --mem=150G                             # Total memory for the job
#SBATCH --time=10:00:00                       

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

# Run the PCA Python script with the arguments passed from sbatch
python /project/3022057.01/IFA/run_IFA/Partial/run_pca_partial.py "$1" "$2" "$3"|| { echo "Failed to run PCA script"; exit 1; }

echo "PCA job completed successfully."