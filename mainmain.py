import os
import json
import argparse

from main import Super_human
from data_util import prepare_test_pp
from util import create_features_dict

protected_map = {
    'Adult': {2: "Female", 1: "Male"},
    'COMPAS': {1: 'Caucasian', 0: 'African-American'},
    'Diabetes': {2: "Female", 1: "Male"},
    'acs_west_poverty': {0: 'White', 1: 'Black'},
    'acs_west_mobility': {0: 'White', 1: 'Black'},
    'acs_west_income': {0: 'White', 1: 'Black'},
    'acs_west_insurance': {0: 'White', 1: 'Black'},
    'acs_west_public': {0: 'White', 1: 'Black'},
    'acs_west_travel': {0: 'White', 1: 'Black'},
    'acs_west_employment': {0: 'White', 1: 'Black'}
}


def parse_args(f):
    with open(f, 'r') as f:
        args = json.load(f)
    return args

noise_list = [0.2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Super Human Fairness")
    parser.add_argument("-c","--config", help="Specify the config file")
    args = parser.parse_args()
    conf = parse_args(args.config)
    # parameters
    dataset = conf['dataset']
    feature_list = conf['features']
    demo_baseline = conf['demo_baseline']
    base_model_type = conf['base_model_type']
    noise = conf['noise']
    num_of_demos = conf['num_of_demos']
    lr_theta = conf['lr_theta']
    noise_ratio = conf['noise_ratio']
    iters = conf['iters']
    alpha = conf['alpha']
    beta = conf['beta']
    task = conf['task']
    model = conf['model']
    sensitive_attribute = conf['sensitive_attribute']
    label = conf['label']
    dict_map = protected_map[dataset]
    
    if noise==False:
        root = "experiments"
    else:
        root = "experiments/noise"
        
    data_path = os.path.join(root, "data")
    model_path = os.path.join(root, "model")
    train_data_path = os.path.join(root, "train")
    test_data_path = os.path.join(root, "test")
    plots_path = os.path.join(root, "plots")
    dataset_path = os.path.join("dataset", dataset, "dataset_ref.csv")
    
    feature, num_of_features = create_features_dict(feature_list)

    if noise==False:
        noise_ratio = 0.0
    sh_obj = Super_human(dataset, num_of_demos, feature, num_of_features, lr_theta, noise, noise_ratio, demo_baseline, base_model_type)
    if noise==False:
        noise_ratio = 0.0

    if task == 'prepare-demos':
        prepare_test_pp(dataset, dataset_path, sensitive_attribute, label, feature, num_of_features,
                        dict_map, demo_baseline,lr_theta, num_of_demos, noise_ratio, train_data_path,
                        test_data_path, data_path,noise=noise, model="logistic_regression", alpha=0.5, beta=0.5)
        
        sh_obj.base_model()

    elif task == 'train':
        print("lr_theta: ", lr_theta)
        sh_obj.base_model()
        sh_obj.update_model(lr_theta, iters)
        sh_obj.test_model(-1)

    elif task == 'test':
        # sh_obj.base_model()
        sh_obj.read_model_from_file()
        # sh_obj.read_nn_model()
        sh_obj.test_model(-1)

    elif task == 'noise-test':
        noise = True
        for noise_ratio in noise_list:
            sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta)
            sh_obj.base_model()
            sh_obj.read_demo_list()
            sh_obj.update_model(lr_theta, iters)
            sh_obj.read_model_from_file()
            sh_obj.test_model(-1)

    elif task == 'test-errorbars':
        for exp_idx in range(conf['num_experiment']):
            print("exp_idx: ", exp_idx)
            sh_obj.prepare_test_pp(model = model, alpha = alpha, beta = beta)
            sh_obj.base_model()
            sh_obj.read_demo_list()
            sh_obj.update_model(lr_theta, iters)
            sh_obj.read_model_from_file()
            sh_obj.test_model(exp_idx)