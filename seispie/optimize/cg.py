from seispie.optimize.base import base
from seispie.optimize.line_search.bracket import bracket
import numpy as np

class cg(base):
	def setup(self, workflow):
		self.solver = workflow.solver
		self.bracket = bracket(workflow.solver, self.config)
		if self.mpi and self.mpi.rank():
			self.bracket.head = 0
		else:
			self.bracket.head = 1

	def line_search(self, misfit):
		return self.bracket.run(self.m_new, self.g_new, self.p_new, misfit)

	def restart_search(self):
		self.line_search.restart()

	def pollak(self, g_new, g_old):
		num = np.dot(g_new, g_new - g_old)
		den = np.dot(g_old, g_old)
		beta = num / den
		return beta

	def compute_direction(self):
		g_new = self.g_new

		if not hasattr(self, 'g_old'):
			return -g_new

		g_old = self.g_old
		p_old = self.p_old

		beta = self.pollak(g_new, g_old)
		p_new = -g_new + beta * p_old

		if np.dot(p_new, g_new) > 0:
			print('  restarting CG: not descent')
			self.restart_search()
			return -g_new

		return p_new

			
