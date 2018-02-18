
import ipdb as pdb
import sys
import yaml

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_type = config_type
        self.config_file = config_file
        self.init_type, self.lr0, self.batch_size, self.num_epochs, self.num_neorons_l1, self.num_neorons_l2, self.filename, self.subsample_ratio, self.config= self.parse_cofig_file()
        print (self.get_config_info())

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
            init_type = configs[self.config_type]['init_type']
            lr0 = configs[self.config_type]['lr0']
            batch_size = configs[self.config_type]['batch_size']
            num_epochs = configs[self.config_type]['num_epochs']
            num_neorons_l1 = configs[self.config_type]['num_neorons_l1']
            num_neorons_l2 = configs[self.config_type]['num_neorons_l2']
            filename = configs[self.config_type]['filename']
            subsample_ratio = configs[self.config_type]['subsample_ratio']
            return init_type, lr0, batch_size, num_epochs, num_neorons_l1, num_neorons_l2, filename, subsample_ratio, configs[self.config_type]
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def get_config_info(self):
        config_str = "initializing weights with: {0} distribution\n"\
                     "initial learning rate is set to: {1}\n"\
                     "batch size is set to: {2} \n"\
                     "number of epochs is set to: {3}\n"\
                     "number of neurons on L1: {4} \n"\
                     "number of neurons on L2: {5} \n"\
                     "output file name prefix: {6} \n"\
                     "subsample ratio: {7} \n".\
                     format(self.init_type, self.lr0, self.batch_size, self.num_epochs, self.num_neorons_l1, self.num_neorons_l2, self.filename, self.subsample_ratio)
        return "========================================================\nConfiguration:\n========================================================\n{0}".format(config_str)

    def get_configs(self, config_type=0):
        if config_type==0:
            return {'init_type': 'zero', 'lr0': 0.01, 'batch_size': 100, 'num_epochs':10}
        elif config_type==1:
            return {'init_type': 'normal', 'lr0': 0.01, 'batch_size': 100, 'num_epochs':10}
        elif config_type==2:
            return {'init_type': 'glorot', 'lr0': 0.01, 'batch_size': 100, 'num_epochs':10}
        elif config_type==3:
            return {'init_type': 'glorot', 'lr0': 0.01, 'batch_size': 100, 'num_epochs':100}
        else:
            print ("Unsupported config type".format(config_type))
            sys.exit()

    def read_config(self):
        configs = None
        with open(self.config_file, 'r') as stream:
            try:
                configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                print ("Error loading YAML file {0}".format(self.config_file))
                sys.exit()
        return configs

