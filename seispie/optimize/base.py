from time import time

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

		for i in range(1):
			print('Iteration %d' % (i+1))
			misfit, self.g_new, self.m_new = solver.compute_gradient()
			solver.export_field(self.m_new, 'mu', i)

			if i == 0:
				ref = misfit

			misfits.append(misfits / ref)
			
			self.p_new = self.compute_direction()
			self.m_old = self.m_new
			self.p_old = self.p_new
			self.g_old = self.g_new

			if self.line_search(misfit) < 0:
				print('Line search failed')
				break

			solver.export_field(self.g_new, 'kmu', i)

		print('')
		print('Misfits: ', misfits)
		print('Elapsed time: %.2f' % (time() - start))

	