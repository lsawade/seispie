from seispie.workflow.base import base
from time import time

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
		solver = self.solver
		start = time()
		solver.import_model(1)
		solver.import_sources()
		solver.import_stations()
		solver.run_forward()
		
		print('elapsed time: %.2fs' % (time() - start))
			

	@property
	def modules(self):
		""" modules to load
		"""
		return ['solver']