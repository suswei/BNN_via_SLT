
----
# Instructions
Install necessary package using [pipenv](https://pipenv.pypa.io/en/latest/):
```
cd /path/to/git/repo/root/
pipenv install .
```
or 
```
pip install -r requirement.txt
```

To run the full set of experiments: 
```
for i in $(seq 0 3839)
do 
 python experiments.py $i
done
```
Alternatively, see `slurmsweep.sh` or `slurmsweep_cpu.sh` SLURM scripts for running in HPC environment.

To collect results into dataframes (as pickled `.pkl` files)
```
python experiments_to_pandas.py --path /path/to/outputdir/
```

To generate plots from the pickled files:
```
python plot_utils.py --output_dirpath "/path/to/output/directory" --datafilepaths /path/to/data1.pkl /path/to/data2.pkl
```