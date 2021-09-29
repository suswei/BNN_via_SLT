import os

# all results are dumped in output directory

########################################################################################################################

os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gamma 2 16 5000 0.5 1000 --display_interval 100 --epochs 10 --seed 1 --viz")

# os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
#           "--var_mode nf_gamma 2 16 500 0.5 100 --display_interval 100 --epochs 10 --seed 1 --viz")
#
# os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 -"
#           "-var_mode nf_gaussian 2 16 5 5e-3 --display_interval 100 --epochs 10 --seed 1 --viz")
#
# os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 -"
#           "-var_mode nf_gaussian 2 16 5 5e-2 --display_interval 100 --epochs 10 --seed 1 --viz")

########################################################################################################################

# os.system("python3 plot_experiments.py --path reducedrank")

# os.system("python3 plot_experiments.py --path tanh")