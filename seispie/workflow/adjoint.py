from seispie.workflow.base import base
from time import time
import numpy as np

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
		solver = self.solver
		solver.import_model(1)
		solver.import_sources()
		solver.import_stations()

		start = time()
		solver.generate_traces()
		mid = time()
		print('')
		print('Trace generation: %.2fs' % (mid - start))
		print('')
		solver.import_model(0)
		solver.run_kernel(1)
		print('')
		print('Elapsed time: %.2fs' % (time() - mid))

		out = np.zeros(solver.nx*solver.nz, dtype='float32')
		solver.k_mu.copy_to_host(out, stream=solver.stream)
		solver.stream.synchronize()
		solver.export_field(out, 'kmu')

	@property
	def modules(self):
		return ['solver', 'postprocess']