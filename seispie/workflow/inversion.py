from . base import base

class inversion:
	def setup(self, config):
		if 'niter' in config:
			self.niter = config['niter']
		else:
			self.niter = 1
		
		if 'slurm' in config:
			self.slurm = config['slurm']
		else:
			self.slurm = 'no'
		
		if 'source_encoding' in config:
			self.source_encoding = config['source_encoding']
		else:
			self.source_encoding = 'no'
		
		if 'save_gradient' in config:
			self.save_gradient = config['save_gradient']
		else:
			self.save_gradient = 'no'
		
		if 'save_misfit' in config:
			self.save_misfit = config['save_misfit']
		else:
			self.save_misfit = 'no'
	
	def main(self):
		print(self.save_gradient)