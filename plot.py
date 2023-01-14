import os
import pickle
from main import Super_human, make_experiment_filename, load_object, find_gamma_superhuman
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


method_label =  {'zehlike':'DELTR', 'policy_learning': 'Fair-PGRank', 'post_processing':'Post_Proc', 'fair_robust': 'Fair_Robust', 'random_ranker': 'Random_Ranker'}
markers = {0:">", 1: "*", 2: "<", 3:"v", 4:"o"}
colors = {0:'orange', 1: 'red', 2: 'blue', 3:'green', 4:'black'}

#feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference", 5: "Positive predictive value difference", 6: "Negative predictive value difference", 7: "Predictive value difference"}
feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "Equalized odds difference", 3: "Predictive value difference"}
name = {0: "Prediction error", 1: "D.DP", 2: "D.EqOdds", 3: "D.PRP"}
short = {"ZeroOne": "error", "Demographic parity difference": "DP", "D.FNR": "FNR", "D.FPR": "FPR", "Equalized odds difference": "EqOdds", "D.PPV": "PPV", "D.NPV":"NPV", "Predictive value difference":"PRP"}
lr_theta = 0.01
num_of_demos = 50
num_of_features = 4
noise_ratio = 0
noise_list = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]#[0.11, 0.12, 0.13, 0.14, 0.15]#[0.05, 0.06, 0.07, 0.08, 0.09, 0.10]#[0.0, 0.01, 0.02, 0.03, 0.04]

def plot_features(noise, dataset):

    if noise==False: noise_ratio = 0.0

    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio)

    experiment_filename = make_experiment_filename(dataset = dataset, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
    file_dir = os.path.join(sh_obj.test_data_path)
    model_params = load_object(file_dir,experiment_filename)


    demo_list = sh_obj.read_demo_list()
    alpha = model_params["alpha"]
    print("alpha: ", alpha)
    alpha = [1 / x for x in alpha]
    print(alpha)
    print(model_params)
    ### TO DO: 1) add legends to the plots
    ###        2) add solid and dashed lines
    ### feature j is on the y axis and i is on the x axis
    find_gamma_superhuman(demo_list, model_params) 

    for i in range(num_of_features):
        for j in range(i+1, num_of_features):
            demo_metric_i = [demo_list[z].metric[i] for z in range(len(demo_list))]
            demo_metric_j = [demo_list[z].metric[j] for z in range(len(demo_list))]
            f1 = plt.figure()
            x = model_params['eval'][-1].loc[feature[j]][0]
            y = model_params['eval'][-1].loc[feature[i]][0]
            x_test = model_params['eval_sh'].loc[feature[j]][0]
            y_test = model_params['eval_sh'].loc[feature[i]][0]
            x_pp_dp = model_params['eval_pp_dp'].loc[feature[j]][0]
            y_pp_dp = model_params['eval_pp_dp'].loc[feature[i]][0]
            x_pp_eq_odds = model_params['eval_pp_eq_odds'].loc[feature[j]][0]
            y_pp_eq_odds = model_params['eval_pp_eq_odds'].loc[feature[i]][0]
            newX = x + alpha[j]
            newY = y + alpha[i]
            xlim = max(max(demo_metric_j), newX)*1.2
            ylim = max(max(demo_metric_i), newY)*1.2
            # xLeft is (xlim - x)*0.8 + x
            yLeft = (ylim - y)*0.8 + y
            xBottom = (xlim - x)*0.8 + x
            plt.xlabel(name[j]) #plt.xlabel(feature[j])
            plt.ylabel(name[i]) #plt.ylabel(feature[i])
            plt.scatter(demo_metric_j, demo_metric_i, marker='*', c=[(255/255,211/255,107/255)], label = 'post_proc_demos')
            plt.plot(x, y, 'ro', label = 'super_human_train')
            plt.plot(x_test, y_test, marker='X', color='black', label = 'super_human_test')
            plt.plot(x_pp_dp, y_pp_dp, 'bo', label = 'post_proc_dp')
            plt.plot(x_pp_eq_odds, y_pp_eq_odds, 'go', label = 'post_proc_eq_odds')
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
            
            handles, labels = plt.gca().get_legend_handles_labels()
            #plt.xlabel('Fairness Violation')
            #plt.ylabel('NDCG')
            plt.grid(True)
            #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
            plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="small")
            plt.title(dataset)
            plots_path_dir = os.path.join(sh_obj.plots_path, short[feature[j]] + "_vs_"+ short[feature[i]] + ".png")
            plt.savefig(plots_path_dir)



def plot_noise_test(dataset):
    noise = True
    feature_gamma = np.zeros((num_of_features, len(noise_list)))

    for noise_idx, noise_ratio in enumerate(noise_list):
        print("noise_idx: ", noise_idx)
        sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio)
        experiment_filename = make_experiment_filename(dataset = dataset, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
        file_dir = os.path.join(sh_obj.test_data_path)
        model_params = load_object(file_dir,experiment_filename)

        demo_list = sh_obj.read_demo_list()
        alpha = model_params["alpha"]
        print("alpha: ", alpha)
        print(model_params)
        gamma_superhuman = find_gamma_superhuman(demo_list, model_params)
        print("feature_gamma: ", feature_gamma)
        print("gamma: ", gamma_superhuman)
        for i in range(num_of_features):
            feature_gamma[i, noise_idx] = gamma_superhuman[i]

    ### Plotting ###
    f1 = plt.figure()
    plt.xlabel('noise ratio') #plt.xlabel(feature[j])
    plt.ylabel('gamma superhumn') #plt.ylabel(feature[i])
    
    for i in range(num_of_features):  
        plt.scatter(noise_list, feature_gamma[i], marker=markers[i], c=colors[i], label = feature[i])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.grid(True)
    plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="small")
    plt.title(dataset)
    plots_path_dir = os.path.join(sh_obj.plots_path, "noise_vs_gamma_superhuman_" + dataset + ".png")
    plt.savefig(plots_path_dir)




        



      



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-t','--task', help='enter the task to do', required=True)
    parser.add_argument('-n','--noise', help='noisy demos used if True', default='False')
    parser.add_argument('-d', '--dataset', help="dataset name", required=True)
    args = vars(parser.parse_args())
    dataset = args['dataset']
    noise = eval(args['noise'])

    if args['task'] == 'normal':
        plot_features(noise, dataset)

    elif args['task'] == 'noise-test':
        plot_noise_test(dataset)

    
