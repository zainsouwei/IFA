#!/bin/bash
#SBATCH --partition=batch                       
#SBATCH --mem=150G                             
#SBATCH --time=2:00:00
#SBATCH --ntasks 1     
#SBATCH --cpus-per-task=15
#SBATCH --output=/project/3022057.01/slurm-%j.out  
#SBATCH --error=/project/3022057.01/slurm-%j.err   

ulimit -v unlimited  # Remove the soft limit on virtual memory
ulimit -d unlimited  # Remove the soft limit on data segment size
ulimit -s unlimited  # Remove the limit on stack size

# Source the environment modules setup script
source /etc/profile.d/modules.sh || { echo "Failed to source environment modules setup script"; exit 1; }

# Load the Anaconda module and activate Conda environment
module load anaconda3 || { echo "Failed to load Anaconda module"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate IFAslurmv2 || { echo "Failed to activate Conda environment"; exit 1; }
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /project/3022057.01/ || { echo "Failed to change directory"; exit 1; }
which python
python --version
# Pass the task argument to the Python script
python /project/3022057.01/IFA/run_IFA/run_parcellate.py "$1"
