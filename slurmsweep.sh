#!/usr/bin/env bash

# The name of the job:
#SBATCH --job-name="tanh_a0"
#SBATCH -p gpgpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpgpuresplat

#SBATCH --mem=5G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-48:0:00

# Batch arrays
#SBATCH --array=0-53

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# Use this email address:
#SBATCH --mail-user=susan.wei@unimelb.edu.au

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)
mkdir -p tanh_a0


# The job command(s):
source activate singularmf
MKL_THREADING_LAYER=GNU python3 experiments.py ${SLURM_ARRAY_TASK_ID}

# python3 in spartan command line
# Python 3.8.5
# import torch; print(torch.__version__)
# 1.8.0.dev20201218