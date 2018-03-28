
import ipdb as pdb
import sys
import yaml
import random


class Configuration():

    def __init__(self, config_type, config_file):
        self.config_file = config_file
        self.config_type = config_type
        self.config_dict =  self.parse_cofig_file()
        self.seq_len

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
            self.seq_len = random.randint(configs[self.config_type]['seq_len_min'], configs[self.config_type]['seq_len_max'])
            return configs[self.config_type]
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def get_config_str(self):
        config_str = "model type: {0} \n"\
                     "initial learning rate is set to: {1}\n"\
                     "number of epochs: {2} \n"\
                     "momentum is set to: {3} \n"\
                     "total number of batches: {4}\n"\
                     "number of hidden units: {5}\n"\
                     "sequence length: {6}\n"\
                     "batch size: {7}\n".\
                     format(self.config_dict['model_type'], self.config_dict['learning_rate'], self.config_dict['num_epochs'], self.config_dict['momentum'],\
                            self.config_dict['num_batches'], self.config_dict['num_hidden'], self.seq_len, self.config_dict['batch_size'])
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

