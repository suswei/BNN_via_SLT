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

os.system("python3 experiments_to_df.py --path reducedrank")

os.system("python3 experiments_to_df.py --path tanh")

########################################################################################################################

os.system("python3 main.py --data tanh 576 5000 True --prior_dist gaussian 5 100 "
          "--var_mode gengamma 2 16 10 1 100 True --display_interval 100 --epochs 2000 --seed 1 --viz")

os.system("python3 main.py --data tanh 576 5000 True --prior_dist gaussian 5 100 "
          "--var_mode gaussian 2 16 1e-1 1e-3 --display_interval 100 --epochs 2000 --seed 1 --viz")

os.system("python3 main.py --data tanh 576 5000 False --prior_dist gaussian 0 100 "
          "--var_mode gengamma 2 16 500 5 100 True --display_interval 100 --epochs 2000 --seed 1 --viz")

os.system("python3 main.py --data tanh 576 5000 False --prior_dist gaussian 0 100 "
          "--var_mode gaussian 2 16 5 5e-2 --display_interval 100 --epochs 2000 --seed 1 --viz")

