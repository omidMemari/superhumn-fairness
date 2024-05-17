import os
import shutil
plot_paths = ['experiments/plots/DP_vs_error_Adult_pp_0-0.pdf', 'experiments/plots/EqOdds_vs_error_Adult_pp_0-0.pdf', 'experiments/plots/PRP_vs_error_Adult_pp_0-0.pdf', 'experiments/plots/EqOdds_vs_DP_Adult_pp_0-0.pdf', 'experiments/plots/PRP_vs_DP_Adult_pp_0-0.pdf', 'experiments/plots/PRP_vs_EqOdds_Adult_pp_0-0.pdf']

folder = '_extract_nn'
# copy files to folder name _extract
if not os.path.exists(folder):
    os.makedirs(folder)
for plot_path in plot_paths:
    shutil.copy2(plot_path, folder)
    
