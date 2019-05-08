from seispie.workflow.base import base

class forward(base):
	""" forward simulation
	"""

	def setup(self, config):
		""" initialize
		"""
		pass
	
	def run(self):
		""" start workflow
		"""
		self.call('solver', 'import_model', self.path['model_true'])
		self.call('solver', 'import_sources', self.path['sources'])
		self.call('solver', 'import_stations', self.path['stations'])
		self.call('solver', 'run_forward')

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']