import os
import pickle
from main import Super_human, make_experiment_filename, load_object
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference"}
short = {"ZeroOne": "0-1", "Demographic parity difference": "DP", "False negative rate difference": "FNR", "False positive rate difference": "FPR", "Equalized odds difference": "EqOdds"}
lr_theta = 0.01
lr_alpha = 0.05
dataset = "Adult"
num_of_demos = 100
num_of_features = 5
#noise = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    #parser.add_argument('-t','--task', help='enter the task to do', required=True)
    parser.add_argument('-n','--noise', help='noisy demos used if True', default=False)
    args = vars(parser.parse_args())

    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, noise = eval(args['noise']))
    experiment_filename = make_experiment_filename(dataset = dataset, lr_theta = lr_theta, lr_alpha = lr_alpha, num_of_demos = num_of_demos)
    file_dir = os.path.join(sh_obj.test_data_path)
    print("file_dir: ", file_dir)
    model_params = load_object(file_dir,experiment_filename)
    demo_list = sh_obj.read_demo_list()
    
    alpha = model_params["alpha"]
    print("alpha: ", alpha)
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
            plt.xlabel(feature[j])
            plt.ylabel(feature[i])
            plt.scatter(demo_metric_j, demo_metric_i, marker='o', c=[(255/255,211/255,107/255)], label = 'demo_list')
            plt.plot(x, y, 'ro', label = 'super_human')
            plt.plot([x, x], [y, ylim], 'r')
            plt.plot([x, xlim], [y, y], 'r')
            plt.plot([newX, newX], [newY, ylim], 'r--')
            plt.plot([newX, xlim], [newY, newY], 'r--')
            plt.annotate('', xy=(newX, yLeft), xytext=(x, yLeft), xycoords='data', textcoords='data',
                        arrowprops={'arrowstyle': '<->'})
            # write the text to the top of the arrow above
            plt.text((newX + x) * 0.5, yLeft, fr"$1/\alpha_{{{short[feature[j]]}}}$", horizontalalignment='center', verticalalignment='bottom')
            plt.annotate('', xy=(xBottom, newY), xytext=(xBottom, y), xycoords='data', textcoords='data',
                        arrowprops={'arrowstyle': '<->'})
            # write the text to the right of the arrow above
            plt.text(xBottom, (newY + y) * 0.5,
                    fr"$1/\alpha_{{{short[feature[i]]}}}$", horizontalalignment='left', verticalalignment='center')
            
            plots_path_dir = os.path.join(sh_obj.plots_path, short[feature[j]] + "_vs_"+ short[feature[i]] + ".png")
            plt.savefig(plots_path_dir)
