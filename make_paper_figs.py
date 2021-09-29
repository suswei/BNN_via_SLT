import os

# all results are dumped in output directory

########################################################################################################################

os.system("python3 main.py --data tanh 121 1000 True --prior_dist gaussian 100 "
          "--var_mode nf_gamma 2 16 100 1 100 --display_interval 100 --epochs 1000 --seed 1 --viz")

os.system("python3 main.py --data tanh 121 1000 True --prior_dist gaussian 100 "
          "--var_mode nf_gaussian 2 16 1 1e-2 --display_interval 100 --epochs 1000 --seed 1 --viz")

os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gamma 2 16 100 1 100 --display_interval 100 --epochs 1000 --seed 1 --viz")

os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gaussian 2 16 1 1e-2 --display_interval 100 --epochs 1000 --seed 1 --viz")

########################################################################################################################

# os.system("python3 plot_experiments.py --path reducedrank")

# os.system("python3 plot_experiments.py --path tanh")