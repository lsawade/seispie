from os import path, getcwd, makedirs
from argparse import ArgumentParser
from configparser import ConfigParser

def get_config():
	# parse arguments
	parser = ArgumentParser()
	parser.add_argument('--workdir', nargs='?', default=getcwd())
	parser.add_argument('--config_file', nargs='?', default='config.ini')
	args = parser.parse_args()

	# assertions
	workdir = args.workdir
	config_file = path.join(workdir, args.config_file)
	assert path.exists(workdir)
	assert path.exists(config_file)

	# parse config.ini
	config = ConfigParser()
	config.read(config_file)
	config['directory']['workdir'] = workdir

	# ensure output directory exists
	outdir = path.join(workdir, config['directory']['output'])
	if not path.exists(outdir):
		print('created directory', outdir)
		makedirs(outdir)


	return config
