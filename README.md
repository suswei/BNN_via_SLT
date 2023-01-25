
----
# Plotting data generated from experiments
Install necessary package using [pipenv](https://pipenv.pypa.io/en/latest/):
```
cd /path/to/git/repo/root/
pipenv install .
```

Generate plots: 
```
python plot_utils.py --output_dirpath "/path/to/output/directory" --datafilepaths /path/to/data1.pkl /path/to/data2.pkl
```