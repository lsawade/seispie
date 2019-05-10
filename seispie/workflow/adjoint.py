from seispie.workflow.base import base
from time import time

class adjoint(base):
	""" forward simulation
	"""

	def setup(self):
		""" initialize
		"""
		pass
	
	def run(self):
		""" start workflow
		"""
		self.solver.import_model(True)
		self.solver.import_sources()
		self.solver.import_stations()

		start = time()
		for i in range(self.solver.nsrc):
			self.solver.taskid = i
			self.solver.run_forward()
			
		print(time() - start)

	@property
	def modules(self):
		return ['solver', 'postprocess']