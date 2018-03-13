
import ipdb as pdb
import sys
import yaml

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_file = config_file
        self.config_type = config_type
        self.num_epochs, self.image_size, self.train_size, self.valid_size, \
        self.test_size, self.batch_size_train, self.batch_size_valid, self.batch_size_test, \
        self.lr0, self.momentum, self.data_set_path, self.model_sample_interval,\
        self.config =  self.parse_cofig_file()
        print (self.get_config_str())

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
            num_epochs = configs[self.config_type]['num_epochs']
            image_size = configs[self.config_type]['image_size']
            train_size = configs[self.config_type]['train_size']
            valid_size = configs[self.config_type]['valid_size']
            test_size = configs[self.config_type]['test_size']
            batch_size_train = configs[self.config_type]['batch_size_train']
            batch_size_valid = configs[self.config_type]['batch_size_valid']
            batch_size_test = configs[self.config_type]['batch_size_test']
            lr0 = configs[self.config_type]['lr0']
            momentum = configs[self.config_type]['momentum']
            data_set_path = configs[self.config_type]['data_set_path']
            model_sample_interval = configs[self.config_type]['model_sample_interval']
            return  num_epochs, image_size, train_size, valid_size, test_size, batch_size_train, \
                    batch_size_valid, batch_size_test, lr0, momentum, data_set_path, \
                    model_sample_interval, configs[self.config_type]
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def get_config_str(self):
        config_str = "initial learning rate is set to: {0}\n"\
                     "number of epochs: {1} \n"\
                     "momentum is set to: {2} \n"\
                     "batch size for training is set to: {3} \n"\
                     "batch size for validation is set to: {4} \n"\
                     "batch size for test is set to: {5} \n".\
                     format(self.lr0, self.num_epochs, self.momentum, self.batch_size_train, self.batch_size_valid, self.batch_size_test)
        return "========================================================\nConfiguration:\n========================================================\n{0}".format(config_str)

    def read_config(self):
        configs = None
        with open(self.config_file, 'r') as stream:
            try:
                configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                print ("Error loading YAML file {0}".format(self.config_file))
                sys.exit()
        return configs

