import os
class data_demo:
    def __init__(self, train_x=None, test_x=None, train_y=None, test_y=None, train_A=None, test_A=None, train_A_str=None, test_A_str=None ,idx_train=None, idx_test=None):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.train_A = train_A
        self.test_A = test_A
        self.train_A_str = train_A_str
        self.test_A_str = test_A_str
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.metric = {}
        
class paths:
    def set_paths(self):
        if self.noise:
            root = "experiments/noise"
        else:
            root = "experiments"
            print("root: ", root)
        self.data_path = os.path.join(root,"data")
        self.model_path = os.path.join(root,"model")
        self.train_data_path = os.path.join(root, "train")
        self.test_data_path = os.path.join(root, "test")
        self.plots_path = os.path.join(root,"plots")
        self.dataset_path = os.path.join("dataset", self.dataset, "dataset_ref.csv")
  