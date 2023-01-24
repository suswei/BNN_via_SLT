#!/usr/bin/env bash

# The name of the job:
#SBATCH --job-name="icml"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

#SBATCH --mem=5G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=4-0:0:00

# Batch arrays
#SBATCH --array=0-3839

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

# The job command(s):
source /usr/local/module/spartan_new.sh
module load gitpython/3.1.14
module load fosscuda/2020b
module load pytorch/1.10.0-python-3.8.6
MKL_THREADING_LAYER=GNU python3 experiments.py ${SLURM_ARRAY_TASK_ID}

