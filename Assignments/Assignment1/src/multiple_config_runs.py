import os
import sys
import argparse
import ipdb as pdb
import config
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=True)
    parser.add_argument('-r', '--runconfigs', help='configurations to run', nargs='+', required=False)
    args = parser.parse_args()
    return vars(args)

def read_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            configs = yaml.load(stream)
        except yaml.YAMLError as exc:
            print ("Error loading YAML file {0}".format(config_file))
            sys.exit()
    return configs

if __name__ == '__main__':
    args = parse_args()
    configs = read_config_file(args['configfile'])
    if args['runconfigs'] is None:
        print ("Please provide the test to run (ex: python multiple_config_runs.py -f config.yaml  -r Q1_1)")
        sys.exit()
    if 'all' in args['runconfigs']:
        print("not yet implemented")
        pass
    else:
        for i in range(0, len(args['runconfigs'])):
            command = "python model.py -f config.yaml -t {0}".format(args['runconfigs'][i])
            os.system(command)