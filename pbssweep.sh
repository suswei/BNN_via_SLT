#!/bin/bash
#PBS -P uj89
#PBS -N neurips
#PBS -M susan.wei@unimelb.edu.au
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=64GB
#PBS -l walltime=48:00:00
#PBS -l wd

# ncpus should be in multiples of 48 for normal queue
# max memory is 64GB per node

module load python3
module load pytorch

for i in {0..71}; do
 python3 experiments.py $i > $PBS_JOBID_$i.log &
done

wait