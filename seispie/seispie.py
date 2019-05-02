from os import path, getcwd
from sys import modules
from importlib import import_module
from argparse import ArgumentParser
from configparser import ConfigParser

def import_object(section, name):
	module = import_module(section + '.' + name);
	return getattr(module, name)()

		
def setup(workdir, config_file):
	# assertions
	config_file = path.join(workdir, config_file)
	assert path.exists(workdir)
	assert path.exists(config_file)

	# parse config.ini
	config = ConfigParser()
	config.read(config_file)

	# create workflow
	workflow = import_object('workflow', config['workflow']['mode'])
	
	# load components
	for section in workflow.modules:
		module = import_object(section, config[section]['method'])
		setattr(workflow, section, module)
	
	# initialize
	workflow.setup(config)

	# start workflow
	workflow.run()


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--workdir', nargs='?', 
        default=getcwd())
	
	parser.add_argument('--config_file', nargs='?', 
        default='config.ini')

	args = parser.parse_args()
	
	setup(args.workdir, args.config_file)
