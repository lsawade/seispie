from seispie.workflow.base import base

class forward(base):
	""" forward simulation
	"""

	def setup(self):
		""" initialize
		"""
		pass
	
	def run(self):
		""" start workflow
		"""
		self.call(0, 'solver', 'import_model', 1)
		self.call(0, 'solver', 'import_sources')
		self.call(0, 'solver', 'import_stations')

		if self.solver.config['combine_sources'] == 'yes':
			self.call(1, 'solver', 'run_forward')
		
		else:
			self.call(self.solver.nsrc, 'solver', 'run_forward')
		
		print('elapsed time:', time() - start)
			

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']