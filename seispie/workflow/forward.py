from . base import base

class forward(base):
	""" forward simulation
	"""

	def setup(self, config):
		""" initialize
		"""
		print(self.solver)

	def run(self):
		""" start workflow
		"""
		print('run')

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']