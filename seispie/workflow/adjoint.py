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
		syn_x = []
		syn_y = []
		syn_z = []
		stream = solver.stream
		nsrc = solver.nsrc
		nrec = solver.nrec
		nt = solver.nt
		
		sh = solver.sh
		psv = solver.psv

		for i in range(nsrc):
			solver.taskid = i
			solver.run_forward()
			if sh:
				out = np.zeros(nt * nrec, dtype='float32')
				solver.obs_y.copy_to_host(out, stream=stream)
				syn_y.append(out)

			if psv:
				out = np.zeros(nt * nrec, dtype='float32')
				solver.obs_x.copy_to_host(out, stream=stream)
				syn_x.append(out)
				out = np.zeros(nt * nrec, dtype='float32')
				solver.obs_z.copy_to_host(out, stream=stream)
				syn_z.append(out)

			stream.synchronize()

		solver.import_model(False)
		solver.setup_adjoint()
		solver.clear_kernels()
		
		for i in range(nsrc):
			solver.taskid = i
			solver.run_forward()
			if sh:
				solver.compute_misfit('y', syn_y[i])
				solver.run_adjoint()

			if psv:
				solver.compute_misfit('x', syn_x[i])
				solver.compute_misfit('z', syn_z[i])
				solver.run_adjoint()

		print('elapsed time:', time() - start)


		out = np.zeros(npt, dtype='float32')
		solver.k_mu.copy_to_host(out, stream)
		stream.synchronize()
		solver.export_field(out, 'kmu')

	@property
	def modules(self):
		return ['solver', 'postprocess']