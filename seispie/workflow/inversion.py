from seispie.workflow.base import base
from time import time
import numpy as np

class inversion(base):
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
		solver.setup()
		solver.import_model(1)
		solver.import_sources()
		solver.import_stations()
		solver.import_traces()
		solver.import_model(0)

		self.optimize.setup(solver)
		self.optimize.solver = solver
		self.optimize.run()

	@property
	def modules(self):
		return ['solver', 'postprocess', 'optimize']