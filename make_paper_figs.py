import os
from dataset_factory import get_lmbda_dim
from experiments import set_sweep_config
import pandas as pd

# all results are dumped in output directory

########################################################################################################################

_, tanh_Hs, rr_Hs = set_sweep_config()

trueRLCT_tanh, dim_tanh = get_lmbda_dim(tanh_Hs, 'tanh')
tanh_pd = pd.DataFrame({'$H$': tanh_Hs, 'RLCT': trueRLCT_tanh, '$d$': dim_tanh})
with open('output/tanh_summary.tex', 'w') as tf:
    tf.write(tanh_pd.to_latex(index=False, escape=False))

trueRLCT_rr, dim_rr = get_lmbda_dim(rr_Hs, 'reducedrank')
rr_pd = pd.DataFrame({'$H$': rr_Hs, 'RLCT': trueRLCT_rr, '$d$': dim_rr})
with open('output/rr_summary.tex', 'w') as tf:
    tf.write(rr_pd.to_latex(index=False, escape=False))

########################################################################################################################

os.system("python3 plot_experiments.py --path reducedrank")

os.system("python3 plot_experiments.py --path tanh")

########################################################################################################################

# os.system("python3 main.py --data tanh 121 1000 True --prior_dist gaussian 100 "
#           "--var_mode nf_gamma 2 16 100 1 100 --display_interval 100 --epochs 1000 --seed 1 --viz")
#
# os.system("python3 main.py --data tanh 121 1000 True --prior_dist gaussian 100 "
#           "--var_mode nf_gaussian 2 16 1 1e-2 --display_interval 100 --epochs 1000 --seed 1 --viz")
#
# os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
#           "--var_mode nf_gamma 2 16 100 1 100 --display_interval 100 --epochs 1000 --seed 1 --viz")
#
# os.system("python3 main.py --data tanh 121 1000 False --prior_dist gaussian 100 "
#           "--var_mode nf_gaussian 2 16 1 1e-2 --display_interval 100 --epochs 1000 --seed 1 --viz")

