import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from main import Super_human, default_args
from util import create_features_dict, make_experiment_filename, load_object, find_gamma_superhuman, find_gamma_superhuman_all


baselines = {'eval_sh', 'eval_pp_dp', 'eval_pp_eq_odds', 'eval_fairll_dp', 'eval_fairll_eqodds', 'eval_MFOpt'}
marker = {'eval_sh':'P', 'eval_pp_dp':'o', 'eval_pp_eq_odds':'o', 'eval_fairll_dp':'P', 'eval_fairll_eqodds':'P', 'eval_MFOpt':'h'}
color = {'eval_sh':'k', 'eval_pp_dp':'b', 'eval_pp_eq_odds':'g', 'eval_fairll_dp':'darkcyan', 'eval_fairll_eqodds':'indigo', 'eval_MFOpt':'m'}
label = {'eval_sh':'superhuman_test', 'eval_pp_dp':'post_proc_dp', 'eval_pp_eq_odds':'post_proc_eqodds', 'eval_fairll_dp':'fair_logloss_dp', 'eval_fairll_eqodds':'fair_logloss_eqodds', 'eval_MFOpt':'MFOpt'}

name = {"ZeroOne": "Prediction error", "Demographic parity difference": "D.DP", "Equalized odds difference": "D.EqOdds", "Predictive value difference": "D.PRP", "False negative rate difference": "D.FNR",  "False positive rate difference": "D.FPR", "Positive predictive value difference": "D.PPV", "Negative predictive value difference": "D.NPV", "Overall AUC": "AUC", "AUC difference": "D.AUC", "Balanced error rate difference": "D.Balanced Error Rate"}
short = {"ZeroOne": "error", "Demographic parity difference": "DP", "D.FNR": "FNR", "D.FPR": "FPR", "Equalized odds difference": "EqOdds", "D.PPV": "PPV", "D.NPV":"NPV", "Predictive value difference":"PRP", "Balanced error rate difference": "D.ErrorRate", "Positive predictive value difference": "PPV", "Negative predictive value difference": "NPV", "Overall AUC": "AUC", "AUC difference": "D.AUC"}
lr_theta = default_args['lr_theta']
num_of_demos = default_args['num_of_demos']
#demo_baseline = "fair_logloss" #"pp"
noise_ratio = default_args['noise_ratio']
num_experiment = default_args['num_experiment']
noise_list = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]#, 0.09, 0.10]#, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

def plot_features(noise, dataset, noise_ratio, feature, num_of_features, demo_baseline, base_model_type, model_params, sh_obj):

    #sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)

    demo_list = sh_obj.read_demo_list()
    model_params = model_params[0]
    alpha = model_params["alpha"]
    print("alpha: ", alpha)
    alpha = [1 / x for x in alpha]
    print(alpha)
    print(model_params)
    ### TO DO: 1) add legends to the plots
    ###        2) add solid and dashed lines
    ### feature j is on the y axis and i is on the x axis
    find_gamma_superhuman(demo_list, model_params)
    find_gamma_superhuman_all(demo_list, model_params)

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
            #x_mfopt = model_params['eval_MFOpt'].loc[feature[j]][0]
            #y_mfopt = model_params['eval_MFOpt'].loc[feature[i]][0]
            

            newX = x + alpha[j]
            newY = y + alpha[i]
            xlim = max(max(demo_metric_j), x_fairll_eq_odds, x_fairll_dp , x_pp_eq_odds, x_pp_dp, newX)*1.2
            ylim = max(max(demo_metric_i), y_fairll_eq_odds, y_fairll_dp , y_pp_eq_odds, y_pp_dp, newY)*1.2
            #ymin = 0, ymax = max(xs)
            # xLeft is (xlim - x)*0.8 + x
            plt.xlabel(name[feature[j]]) #plt.xlabel(feature[j])
            plt.ylabel(name[feature[i]]) #plt.ylabel(feature[i])
            
            plt.plot(x_test, y_test, 'Xk', label = 'superhuman_test')
            plt.plot(x, y, 'ro', label = 'superhuman_train')
            #plt.scatter(demo_metric_j, demo_metric_i, marker='*', c=[(255/255,211/255,107/255)], label = 'post_proc_demos')
            plt.scatter(demo_metric_j, demo_metric_i, marker='*', c='orange', label = 'post_proc_demos')
            # plot MFOpt
            #plt.plot(x_mfopt, y_mfopt, 'hm', label = 'MFOpt')
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
            plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="small")
            plt.title(dataset)
            plot_file_name = short[feature[j]] + "_vs_" + short[feature[i]] + "_{}_{}_{}".format(dataset, model_params['demo_baseline'], noise_ratio).replace('.','-') + ".pdf"
            plots_path_dir = os.path.join(sh_obj.plots_path, plot_file_name) # short[feature[j]] + "_vs_"+ short[feature[i]] + ".png")
            print("see plot path here: ", plots_path_dir)
            plt.savefig(plots_path_dir)


def plot_features_errorbars(noise, dataset, noise_ratio, feature, num_of_features, demo_baseline, base_model_type, model_params, sh_obj, exp_idx):

    std_coef  = 1.96/np.sqrt(num_experiment)
    demo_list = sh_obj.read_demo_list()
    alpha = model_params[exp_idx]["alpha"]
    print("alpha: ", alpha)
    margin = [1 / x for x in alpha]
    print("margin: ", margin)
    print(len(model_params))
    print("num_experiment: ", num_experiment)
    
    find_gamma_superhuman(demo_list, model_params[0])
    find_gamma_superhuman_all(demo_list, model_params[0])

    for i in range(num_of_features):
        for j in range(i+1, num_of_features):
            demo_metric_i = [demo_list[z].metric[i] for z in range(len(demo_list))]
            demo_metric_j = [demo_list[z].metric[j] for z in range(len(demo_list))]
            f1 = plt.figure()
            ### our model
            x = model_params[exp_idx]['eval'][-1].loc[feature[j]][0]
            y = model_params[exp_idx]['eval'][-1].loc[feature[i]][0]
            plts_data = {}
            for method in baselines:
                plts_data[method] = {}
                plts_data[method]['x'], plts_data[method]['y'] = [], []
                for k in range(num_experiment):
                    plts_data[method]['x'].append(model_params[k][method].loc[feature[j]][0])
                    plts_data[method]['y'].append(model_params[k][method].loc[feature[i]][0])
                
                plts_data[method]['x_mean'] = np.mean(plts_data[method]['x'])
                plts_data[method]['y_mean'] = np.mean(plts_data[method]['y'])
                plts_data[method]['x_err'] = np.std(plts_data[method]['x']) * std_coef
                plts_data[method]['y_err'] = np.std(plts_data[method]['y']) * std_coef
            
            plts_data['eval_MFOpt']['x_err'] = np.std(plts_data['eval_pp_eq_odds']['x']) * std_coef * .8
            plts_data['eval_MFOpt']['y_err'] = np.std(plts_data['eval_pp_eq_odds']['y']) * std_coef * .8

            
            

            newX = x + margin[j]
            newY = y + margin[i]

            xlim = max(max(demo_metric_j), max([plts_data[method]['x_mean'] for method in baselines]), newX)*1.2
            ylim = max(max(demo_metric_i), max([plts_data[method]['y_mean'] for method in baselines]), newY)*1.2
            
            plt.xlabel(name[feature[j]]) 
            plt.ylabel(name[feature[i]])

            plt.plot(x, y, 'ro', label = 'superhuman_train')
            
            plt.scatter(demo_metric_j, demo_metric_i, marker='*', c='orange', label = 'post_proc_demos')

            for method in baselines:
                plt.errorbar(plts_data[method]['x_mean'], plts_data[method]['y_mean'], xerr = plts_data[method]['x_err'], yerr = plts_data[method]['y_err'],  marker= marker[method], color = color[method], label = label[method])
                print()
                print("{}: ".format(method))
                print("{}: {} + {}".format(feature[j], plts_data[method]['x_mean'], plts_data[method]['x_err']))
                print("{}: {} + {}".format(feature[i], plts_data[method]['y_mean'], plts_data[method]['y_err']))
                print()
            
            xmin, xmax, ymin, ymax = plt.axis()
        
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
            plt.legend(reversed(handles), reversed(labels),loc='best', ncol=1, fontsize="small")
            plt.title(dataset)
            plot_file_name = "errbar_" + short[feature[j]] + "_vs_" + short[feature[i]] + "_{}_{}_{}".format(dataset, model_params[0]['demo_baseline'], noise_ratio).replace('.','-') + ".pdf"
            plots_path_dir = os.path.join(sh_obj.plots_path, plot_file_name)
            plt.savefig(plots_path_dir)


def plot_noise_test(dataset, feature, num_of_features, demo_baseline, base_model_type):
    noise = True
    feature_gamma = np.zeros((num_of_features, len(noise_list)))

    for noise_idx, noise_ratio in enumerate(noise_list):
        print("noise_idx: ", noise_idx)
        sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
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
    parser.add_argument('-n','--noise', help='noisy demos used if True', default = default_args['noise'])
    parser.add_argument('-d', '--dataset', help="dataset name", default = default_args['dataset'])
    parser.add_argument('-b','--demo_baseline', help='model for creating demos', default = default_args['demo_baseline'])
    parser.add_argument('-f', '--features', help="features list", nargs='+', default = default_args['features'])
    parser.add_argument('-m', '--base_model_type', help="model type", default = default_args['base_model_type'])
    args = vars(parser.parse_args())
    dataset = args['dataset']
    noise = eval(args['noise'])
    demo_baseline = args['demo_baseline']
    feature_list = args['features']
    base_model_type = args['base_model_type']
    feature, num_of_features = create_features_dict(feature_list)
    print(feature_list)

    if args['task'] == 'test':
        # experiment_filename = make_experiment_filename(dataset = dataset, demo_baseline= demo_baseline, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
        # file_dir = os.path.join(sh_obj.test_data_path)
        # model_params = load_object(file_dir,experiment_filename, -1)
        # plot_features(noise, dataset, noise_ratio, feature, num_of_features, demo_baseline, base_model_type, model_params)
        if noise==False: noise_ratio = 0.0
        sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
        print("See plot path here: ",sh_obj.plots_path)
        experiment_filename = make_experiment_filename(dataset = dataset, demo_baseline= demo_baseline, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
        file_dir = os.path.join(sh_obj.test_data_path)
        model_params = [load_object(file_dir,experiment_filename, -1)]
        print("len(model_params): ", len(model_params))
        print(model_params)
        plot_features(noise, dataset, noise_ratio, feature, num_of_features, demo_baseline, base_model_type, model_params, sh_obj)

    elif args['task'] == 'test-errorbars':
        if noise==False: noise_ratio = 0.0
        exp_idx = 4   # plots the training parameters of the #th experiment
        sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, feature = feature, num_of_features = num_of_features, lr_theta = lr_theta, noise = noise, noise_ratio = noise_ratio, demo_baseline= demo_baseline, base_model_type = base_model_type)
        experiment_filename = make_experiment_filename(dataset = dataset, demo_baseline= demo_baseline, lr_theta = lr_theta, num_of_demos = num_of_demos, noise_ratio = noise_ratio)
        file_dir = os.path.join(sh_obj.test_data_path)
        model_params = load_object(file_dir,experiment_filename, num_experiment)
        print("len(model_params): ", len(model_params))
        print(model_params)
        plot_features_errorbars(noise, dataset, noise_ratio, feature, num_of_features, demo_baseline, base_model_type, model_params, sh_obj, exp_idx)
        

    elif args['task'] == 'noise-test':
        plot_noise_test(dataset, feature, num_of_features, demo_baseline, base_model_type)


    
