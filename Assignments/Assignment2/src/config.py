import sys
import yaml

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_type = config_type
        self.config_file = config_file
        self.model_type, self.init_type, self.lr0, self.batch_size, self.num_epochs, self.weight_decay, self.batch_norm, self.fig_caption= self.parse_config_file()
        
    def parse_config_file(self):
        configs = self.read_config()
        try:
            model_type = configs[self.config_type]['model_type']
            init_type = configs[self.config_type]['init_type']
            lr0 = configs[self.config_type]['lr0']
            batch_size = configs[self.config_type]['batch_size']
            num_epochs = configs[self.config_type]['num_epochs']
            weight_decay = configs[self.config_type]['weight_decay']
            batch_norm = configs[self.config_type]['batch_norm']
            fig_caption = configs[self.config_type]['fig_caption']
            return model_type, init_type, lr0, batch_size, num_epochs, weight_decay, batch_norm, fig_caption
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def __str__(self):
        config_str = "config type: {} \n"\
                    "model type: {} \n"\
                    "weight initialisation: {} \n"\
                    "learning rate: {} \n"\
                    "batch size: {} \n"\
                    "number of epochs: {}\n"\
                    "weight decay: {} \n"\
                    "batch norm: {} \n"\
                    "fig caption: {} \n"
        return (config_str.format(self.config_type, self.model_type, self.init_type, self.lr0, self.batch_size, self.num_epochs, self.weight_decay, self.batch_norm, self.fig_caption))


    def read_config(self):
        configs = None
        with open(self.config_file, 'r') as stream:
            try:
                configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                print ("Error loading YAML file {0}".format(self.config_file))
                sys.exit()
        return configs