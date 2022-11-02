
import os
from main import Super_human, make_experiment_filename, load_object

# read files in the experiment/test
# read demo_list.pickle
f = open("./experiments/test/Adult_0-05_0-05_100")

root = "experiments"
test_path = os.path.join(root,"test")

# plot zeroOne vs each fairness violations (0-1 vs dp, 0-1 vs eqOdds, ...) for both experiment/test and demo_list.pickle in one plot

lr_theta = 0.05
lr_alpha = 0.05
dataset = "Adult"
num_of_demos = 100
num_of_features = 5

def read_demo_list():
    with open('demo_list.pickle', 'rb') as handle:
      demo_list = pickle.load(handle)
    return demo_list


# look at read_model_from_file(self) from main.py to get an idea of how to read experiment file and base_model.pickel and demo_list.pickle

experiment_filename = make_experiment_filename(dataset = dataset, lr_theta = lr_theta, lr_alpha = lr_alpha, num_of_demos = num_of_demos)
file_dir = os.path.join(test_path)
model_params = load_object(file_dir,experiment_filename)

print(model_params)

sh_obj = Super_human(dataset = dataset, num_of_demos = num_of_demos, num_of_features = num_of_features)

demo_list = sh_obj.read_demo_list()

print(demo_list[0].metric)

print(model_params['eval_sh'].loc['Demographic parity difference'])
    