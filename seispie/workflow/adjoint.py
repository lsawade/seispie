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

		solver.import_model(True)
		solver.import_sources()
		solver.import_stations()

		start = time()
		syn_x = solver.syn_x = []
		syn_y = solver.syn_y = []
		syn_z = solver.syn_z = []
		stream = solver.stream
		nsrc = solver.nsrc
		nrec = solver.nrec
		nt = solver.nt
		
		sh = 1 if solver.config['sh'] == 'yes' else 0
		psv = 1 if solver.config['psv'] == 'yes' else 0

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

		for i in range(nsrc):
			solver.taskid = i
			solver.run_forward()
			solver.setup_adjoint()

			# import matplotlib.pyplot as plt
			# plt.plot(obs[i][0:solver.nt])
			# plt.plot(out[i:solver.nt])
			# plt.show()

		print(time() - start)

	@property
	def modules(self):
		return ['solver', 'postprocess']