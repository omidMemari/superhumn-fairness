
# read files in the experiment/test
# read demo_list.pickle

# plot zeroOne vs each fairness violations (0-1 vs dp, 0-1 vs eqOdds, ...) for both experiment/test and demo_list.pickle in one plot

lr_theta = 0.05
lr_alpha = 0.05
dataset = "Adult"
num_of_demos = 100
num_of_features = 5

# look at read_model_from_file(self) from main.py to get an idea of how to read experiment file and base_model.pickel and demo_list.pickle
