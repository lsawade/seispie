from time import time
import numpy as np

class base:
	def setup(self):
		raise NotImplementedError

	def compute_direction(self):
		raise NotImplementedError

	def line_search(self, misfit):
		raise NotImplementedError

	def restart_search(self):
		raise NotImplementedError

	def run(self):
		solver = self.solver
		niter = int(self.config['niter'])
		start = time()

		misfits = []

		for i in range(niter):
			print('Iteration %d' % (i+1))
			misfit, self.g_new, self.m_new = solver.compute_gradient()
			
			if i > 0:
				solver.export_field(self.m_new, 'mu', i-1)
			else:
				ref = misfit

			misfits.append(misfit / ref)
			
			self.p_new = self.compute_direction()
			self.m_old = self.m_new
			self.p_old = self.p_new
			self.g_old = self.g_new

			if self.line_search(misfit) < 0:
				print('Line search failed')
				break

		misfit = solver.compute_misfit()
		misfits.append(misfit / ref)
		m_new = np.zeros(solver.nx * solver.nz, dtype='float32')
		solver.mu.copy_to_host(m_new, solver.stream)
		solver.stream.synchronize()
		solver.export_field(m_new, 'mu', niter)

		print('')
		print('Misfits:')
		for misfit in misfits:
			print('  %.2f', misfit)

		print('')
		print('Elapsed time: %.2f' % (time() - start))

	