from . base import base

class forward(base):
	""" forward simulation
	"""

	def setup(self, config):
		""" initialize
		"""
	
	def run(self):
		""" start workflow
		"""
		self.call_mpi()

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']