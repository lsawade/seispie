from . base import base

class forward(base):
	""" forward simulation
	"""

	def setup(self, config):
		""" initialize
		"""
		self.niter = config['niter']
	
	def run(self):
		""" start workflow
		"""
		self.call()

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']