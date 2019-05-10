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
		self.call(0, 'solver', 'import_model', self.path['model_true'])
		self.call(0, 'solver', 'import_sources', self.path['sources'])
		self.call(0, 'solver', 'import_stations', self.path['stations'])

		print(self.solver.config['combine_sources'] )
		if self.solver.config['combine_sources'] == 'yes':
			self.call(1, 'solver', 'run_forward')
			# self.call(1, 'solver', 'export_traces')
			# self.call(1, 'solver', 'export_snapshots')
		
		else:
			self.call(self.solver.nsrc, 'solver', 'run_forward')
			# self.call(self.nsrc, 'solver', 'export_traces')
			# self.call(self.nsrc, 'solver', 'export_snapshots')
			

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']