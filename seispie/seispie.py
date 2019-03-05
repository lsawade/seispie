from os import path, getcwd
from sys import modules
from importlib import import_module
import argparse, configparser
		
def setup(workdir, config_file):
	names = [
		'workflow',
		'preprocess',
		'solver',
		'postprocess',
		'optimize'
	]

	config = configparser.ConfigParser()
	config.read(path.join(workdir, config_file))
	modules['seispie_config'] = config

	for name in names:
		module = import_module(name + '.' + config[name][name])
		modules['seispie_' + name] = getattr(module, config[name][name])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--workdir', nargs='?', 
        default=getcwd())
	
	parser.add_argument('--config_file', nargs='?', 
        default='config.ini')

	args = parser.parse_args()
	
	setup(args.workdir, args.config_file)
