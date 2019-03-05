from sys import modules
from . base import base

class forward:
	def setup(self, config):
		if 'slurm' in config:
			self.slurm = config['slurm']
		else:
			self.slurm = 'no'
	
	def main(self):
		solver = modules['seispie_solver']
		cfg = modules['seispie_config']['path']
		solver.importModel(cfg['model_true'])
		solver.runForward()