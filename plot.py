import os
import pickle
from main import Super_human, make_experiment_filename, load_object
import matplotlib.pyplot as plt
import seaborn as sns

feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference"}
lr_theta = 0.05
lr_alpha = 0.05
dataset = "Adult"
num_of_demos = 100
num_of_features = 5
noise = False

sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, noise = noise)
experiment_filename = make_experiment_filename(dataset = dataset, lr_theta = lr_theta, lr_alpha = lr_alpha, num_of_demos = num_of_demos)
file_dir = os.path.join(sh_obj.test_data_path)
print("file_dir: ", file_dir)
model_params = load_object(file_dir,experiment_filename)
demo_list = sh_obj.read_demo_list()

### TO DO: 1) add legends to the plots
###        2) add solid and dashed lines

for i in range(num_of_features):
    for j in range(i+1, num_of_features):
        demo_metric_i = [demo_list[z].metric[i] for z in range(len(demo_list))]
        demo_metric_j = [demo_list[z].metric[j] for z in range(len(demo_list))]
        f1 = plt.figure()
        plt.plot(demo_metric_j, demo_metric_i, 'go', label = 'demo_list')
        plt.plot(model_params['eval_sh'].loc[feature[j]], model_params['eval_sh'].loc[feature[i]], 'ro', label = 'super_human')
        plots_path_dir = os.path.join(sh_obj.plots_path, feature[j] + "_vs_"+ feature[i] + ".png")
        plt.savefig(plots_path_dir)


# zero_ones = []
# dp = []
# fnr = []
# fpr = []
# eq_odds = []

# for i in range(len(demo_list)):
#     zero_ones.append(demo_list[i].metric[0])
#     dp.append(demo_list[i].metric[1])
#     fnr.append(demo_list[i].metric[2])
#     fpr.append(demo_list[i].metric[3])
#     eq_odds.append(demo_list[i].metric[4])
    
# print(model_params['eval_sh'])
# zero_one_sh = model_params['eval_sh'].loc[feature[0]]
# dp_sh = model_params['eval_sh'].loc[feature[1]]
# fnr_sh = model_params['eval_sh'].loc[feature[2]]
# fpr_sh = model_params['eval_sh'].loc[feature[3]]
# eq_odds_sh = model_params['eval_sh'].loc[feature[4]]


# # plot zero_ones as y-axis and dp as x-axis in green color
# # plot zero_one_sh as y-axis and dp_sh as x-axis in red color
# plt.plot(dp, zero_ones, 'go', label = 'demo_list')
# plt.plot(dp_sh, zero_one_sh, 'ro', label = 'sh')
# plt.savefig("zeroOne vs Dp.png")

# plt.plot(eq_odds, zero_ones, 'go', label = 'demo_list')
# plt.plot(eq_odds_sh, zero_one_sh, 'ro', label = 'sh')
# plt.savefig("zeroOne vs EqOdds.png")

# plt.plot(eq_odds, zero_ones, 'go', label = 'demo_list')
# plt.plot(eq_odds_sh, zero_one_sh, 'ro', label = 'sh')
# plt.savefig("zeroOne vs EqOdds.png")