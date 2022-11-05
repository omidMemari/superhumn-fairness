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

alpha = model_params["alpha"]
alpha = [1 / x for x in alpha]
print(alpha)
### TO DO: 1) add legends to the plots
###        2) add solid and dashed lines
### feature j is on the y axis and i is on the x axis

for i in range(num_of_features):
    for j in range(i+1, num_of_features):
        demo_metric_i = [demo_list[z].metric[i] for z in range(len(demo_list))]
        demo_metric_j = [demo_list[z].metric[j] for z in range(len(demo_list))]
        f1 = plt.figure()
        x = model_params['eval_sh'].loc[feature[j]][0]
        y = model_params['eval_sh'].loc[feature[i]][0]
        newX = x + alpha[j]
        newY = y + alpha[i]
        print(feature[j], " vs " ,feature[i])
        print("x: ", x , "y: ", y)
        print("newX: ", newX , "newY: ", newY)
        print()
        xlim = 0.5 #max(max(demo_metric_j), x, newX)
        ylim = 0.5 #max(max(demo_metric_i), y, newY)
        # xLeft is (xlim - x)*0.8 + x
        yLeft = (ylim - y)*0.8 + y
        xBottom = (xlim - x)*0.8 + x
        plt.plot(demo_metric_j, demo_metric_i, 'go', label = 'demo_list')
        plt.plot(x, y, 'ro', label = 'super_human')
        plt.plot([x, x], [y, ylim], 'r')
        plt.plot([x, xlim], [y, y], 'r')
        plt.plot([newX, newX], [newY, ylim], 'r--')
        plt.plot([newX, xlim], [newY, newY], 'r--')
        plt.annotate('', xy=(newX, yLeft), xytext=(x, yLeft), xycoords='data', textcoords='data',
                     arrowprops={'arrowstyle': '<->'})
        # write the text to the top of the arrow above
        plt.text((newX + x) * 0.5, yLeft + yLeft*0.01, r'$\alpha_j$', horizontalalignment='center', verticalalignment='center')
        plt.annotate('', xy=(xBottom, newY), xytext=(xBottom, y), xycoords='data', textcoords='data',
                     arrowprops={'arrowstyle': '<->'})
        # write the text to the right of the arrow above
        plt.text(xBottom + xBottom*0.03, (newY + y) * 0.5, r'$\alpha_i$', horizontalalignment='center', verticalalignment='center')
        
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