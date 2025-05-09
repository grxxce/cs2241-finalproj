#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-02:00       # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test         # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -J nosub_pu1k_05radius_patch_uniform_to_uniform  # Job name


# 1) pu1k_05radius_patch_uniform_to_fps
# 2) pu1k_05radius_patch_poisson_to_fps
# 3) pu1k_05radius_patch_poisson_to_uniform
# 4) pu1k_05radius_patch_uniform_to_uniform



# Initialize conda directly from your miniconda3 installation
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate pugcn-fresh

# Print environment info for debugging
echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Run your code
python Training.py 