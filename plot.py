import os
import pickle
from main import Super_human, make_experiment_filename, load_object, find_gamma_superhuman
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


#method_label =  {'zehlike':'DELTR', 'policy_learning': 'Fair-PGRank', 'post_processing':'Post_Proc', 'fair_robust': 'Fair_Robust', 'random_ranker': 'Random_Ranker'}
markers = {0:"o", 1: "*", 2: "x", 3:"<", 4:"v"}
colors = {0:'orange', 1: 'red', 2: 'blue', 3:'green', 4:'black'}

#feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "False negative rate difference", 3: "False positive rate difference", 4: "Equalized odds difference", 5: "Positive predictive value difference", 6: "Negative predictive value difference", 7: "Predictive value difference"}
feature = {0: "ZeroOne", 1: "Demographic parity difference", 2: "Equalized odds difference", 3: "Predictive value difference"}
name = {0: "Prediction error", 1: "D.DP", 2: "D.EqOdds", 3: "D.PRP"}
short = {"ZeroOne": "error", "Demographic parity difference": "DP", "D.FNR": "FNR", "D.FPR": "FPR", "Equalized odds difference": "EqOdds", "D.PPV": "PPV", "D.NPV":"NPV", "Predictive value difference":"PRP"}
lr_theta = 0.03
num_of_demos = 50
num_of_features = 4
demo_baseline = "pp"#"fair_logloss"
noise_ratio = 0.2
noise_list = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]#, 0.09, 0.10]#, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

def plot_features(noise, dataset, noise_ratio):

    if noise==False: noise_ratio = 0.0

    sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio)

    experiment_filename = make_experiment_filename(dataset = dataset, demo_baseline= demo_baseline, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
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
            ### our model
            x = model_params['eval'][-1].loc[feature[j]][0]
            y = model_params['eval'][-1].loc[feature[i]][0]
            x_test = model_params['eval_sh'].loc[feature[j]][0]
            y_test = model_params['eval_sh'].loc[feature[i]][0]
            ### post-processing
            x_pp_dp = model_params['eval_pp_dp'].loc[feature[j]][0]
            y_pp_dp = model_params['eval_pp_dp'].loc[feature[i]][0]
            x_pp_eq_odds = model_params['eval_pp_eq_odds'].loc[feature[j]][0]
            y_pp_eq_odds = model_params['eval_pp_eq_odds'].loc[feature[i]][0]
            ### fair log-loss
            x_fairll_dp = model_params['eval_fairll_dp'].loc[feature[j]][0]
            y_fairll_dp = model_params['eval_fairll_dp'].loc[feature[i]][0]
            x_fairll_eq_odds = model_params['eval_fairll_eqodds'].loc[feature[j]][0]
            y_fairll_eq_odds = model_params['eval_fairll_eqodds'].loc[feature[i]][0]
            x_fairll_eq_opp = model_params['eval_fairll_eqopp'].loc[feature[j]][0]
            y_fairll_eq_opp = model_params['eval_fairll_eqopp'].loc[feature[i]][0]
            ### MFOpt
            x_mfopt = model_params['eval_MFOpt'].loc[feature[j]][0]
            y_mfopt = model_params['eval_MFOpt'].loc[feature[i]][0]
            

            newX = x + alpha[j]
            newY = y + alpha[i]
            xlim = max(max(demo_metric_j), x_fairll_eq_odds, x_fairll_dp , x_pp_eq_odds, x_pp_dp, newX)*1.2
            ylim = max(max(demo_metric_i), y_fairll_eq_odds, y_fairll_dp , y_pp_eq_odds, y_pp_dp, newY)*1.2
            #ymin = 0, ymax = max(xs)
            # xLeft is (xlim - x)*0.8 + x
            plt.xlabel(name[j]) #plt.xlabel(feature[j])
            plt.ylabel(name[i]) #plt.ylabel(feature[i])
            
            plt.plot(x_test, y_test, 'Xk', label = 'superhuman_test')
            plt.plot(x, y, 'ro', label = 'superhuman_train')
            #plt.scatter(demo_metric_j, demo_metric_i, marker='*', c=[(255/255,211/255,107/255)], label = 'post_proc_demos')
            plt.scatter(demo_metric_j, demo_metric_i, marker='*', c='orange', label = 'post_proc_demos')
            # plot MFOpt
            plt.plot(x_mfopt, y_mfopt, 'hm', label = 'MFOpt')
            # plot post-processing
            plt.plot(x_pp_dp, y_pp_dp, 'bo', label = 'post_proc_dp')
            plt.plot(x_pp_eq_odds, y_pp_eq_odds, 'go', label = 'post_proc_eqodds')
            # plot fair log-loss
            plt.plot(x_fairll_dp, y_fairll_dp, marker='P', color='darkcyan', label = 'fair_logloss_dp')
            plt.plot(x_fairll_eq_odds, y_fairll_eq_odds, marker='P', color='indigo', label = 'fair_logloss_eqodds')
            #plt.plot(x_fairll_dp, y_fairll_eq_opp, marker='P', color='cyan', label = 'fair_logloss_eqopp')
            
            xmin, xmax, ymin, ymax = plt.axis()
            
            # plt.plot([x, x], [y, 0.95*ymax], 'r')
            # plt.plot([x, 0.95*xmax], [y, y], 'r')
            # plt.plot([newX, newX], [newY, 0.95*ymax], 'r--')
            # plt.plot([newX, 0.95*xmax], [newY, newY], 'r--')
            # yLeft = (ymax - y)*0.8 + y
            # xBottom = (ymax - x)*0.8 + x
            yLeft = (ylim - y)*0.8 + y
            xBottom = (xlim - x)*0.8 + x
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
            plt.grid(True)
            #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
            #plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="medium")
            plt.title(dataset)
            plot_file_name = short[feature[j]] + "_vs_" + short[feature[i]] + "_{}_{}_{}".format(dataset, model_params['demo_baseline'], noise_ratio).replace('.','-') + ".pdf"
            plots_path_dir = os.path.join(sh_obj.plots_path, plot_file_name) # short[feature[j]] + "_vs_"+ short[feature[i]] + ".png")
            plt.savefig(plots_path_dir)

            # get handles and labels for reuse
            #label_params = ax.get_legend_handles_labels() 

            figl, axl = plt.subplots(figsize=(16,1))
            axl.axis(False)
            axl.legend(reversed(handles), reversed(labels), loc="center", ncols=8, prop={"size":11}, fontsize="large")
            plot_file_name1 = "legend.pdf" #short[feature[j]] + "_vs_" + short[feature[i]] + "_{}_{}_{}".format(dataset, model_params['demo_baseline'], noise_ratio).replace('.','-') + ".png"
            plots_path_dir1 = os.path.join(sh_obj.plots_path, plot_file_name1) # short[feature[j]] + "_vs_"+ short[feature[i]] + ".png")
            figl.savefig(plots_path_dir1)



def plot_noise_test(dataset):
    noise = True
    feature_gamma = np.zeros((num_of_features, len(noise_list)))

    for noise_idx, noise_ratio in enumerate(noise_list):
        print("noise_idx: ", noise_idx)
        sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio)
        experiment_filename = make_experiment_filename(dataset = dataset, demo_baseline = demo_baseline, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
        #experiment_filename = "{}_{}_{}_{}".format(dataset, lr_theta, num_of_demos, noise_ratio).replace('.','-')
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
    plt.xlabel('Noise Ratio')
    plt.ylabel(r'$\gamma$-Superhumn')
    overlapping = {0: 0.8, 1:0.7, 2:0.4, 3:0.4}
    line_width = {0: 5, 1: 4, 2: 3, 3: 3}
    for i in range(num_of_features):  
        plt.plot(noise_list, feature_gamma[i], marker=markers[i], c=colors[i], label = feature[i], alpha=overlapping[i], lw=line_width[i])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.grid(True)
    plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="small")
    plt.title(dataset)
    plots_path_dir = os.path.join(sh_obj.plots_path, "noise_vs_gamma_superhuman_" +"_{}_{}".format(dataset, demo_baseline) + ".png")
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
        plot_features(noise, dataset, noise_ratio)

    elif args['task'] == 'noise-test':
        plot_noise_test(dataset)

    
