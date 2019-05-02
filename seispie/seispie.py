from os import path, getcwd
from sys import modules
from importlib import import_module
from argparse import ArgumentParser
from configparser import ConfigParser
		
def setup(workdir, config_file):
	# FIXME remove sys.modules
	names = [
		'workflow',
		'preprocess',
		'solver',
		'postprocess',
		'optimize'
	]

	config = ConfigParser()
	config.read(path.join(workdir, config_file))
	for key in config['path']:
		config['path'][key] = path.join(workdir, config['path'][key])
	
	modules['seispie_config'] = config

	for name in names:
		module = import_module(name + '.' + config[name][name])
		module_target = getattr(module, config[name][name])()
		module_target.setup(config[name])
		modules['seispie_' + name] = module_target

	modules['seispie_workflow'].main()


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--workdir', nargs='?', 
        default=getcwd())
	
	parser.add_argument('--config_file', nargs='?', 
        default='config.ini')

	args = parser.parse_args()
	
	setup(args.workdir, args.config_file)
