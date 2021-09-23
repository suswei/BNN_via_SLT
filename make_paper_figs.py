import os

# all results are dumped in output directory

os.system("python3 main.py --data tanh 1 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gamma 2 16 1000 1 --display_interval 100 --epochs 500 --seed 2 --viz")

os.system("python3 main.py --data tanh 1 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gaussian 2 16 0 1 --display_interval 100 --epochs 500 --seed 2 --viz")

os.system("python3 main.py --data tanh 256 1000 False --prior_dist gaussian 100 "
          "--var_mode nf_gamma 2 16 1000 1 --display_interval 100 --epochs 500 --seed 1 --viz")

os.system("python3 main.py --data tanh 256 1000 False --prior_dist gaussian 100 -"
          "-var_mode nf_gaussian 2 16 0 1 --display_interval 100 --epochs 500 --seed 1 --viz")

os.system("python3 plot_experiments.py --path reducedrank --savefig")
os.system("python3 plot_experiments.py --path tanh --savefig")