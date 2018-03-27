
import ipdb as pdb
import sys
import yaml

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_file = config_file
        self.config_type = config_type
        self.config =  self.parse_cofig_file()
        print (self.get_config_str())

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
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
                     "number of hidden units: {5}\n".\
                     format(self.config['model_type'], self.config['learning_rate'], self.config['num_epochs'], self.config['momentum'],\
                            self.config['num_batches'], self.config['num_hidden'])
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

