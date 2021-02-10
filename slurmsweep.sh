#!/usr/bin/env bash

# The name of the job:
#SBATCH --job-name="highHtanh"
#SBATCH -p physical

#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-12:0:00

# Batch arrays
#SBATCH --array=60-104

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
source activate singularmf
python3 sweep.py ${SLURM_ARRAY_TASK_ID}
