from seispie.optimize.cg import cg
from seispie.optimize.line_search.backtrack import backtrack
import numpy as np

class lbfgs(cg):
	def setup(self, workflow):
		self.solver = workflow.solver
		self.bracket = backtrack(workflow.solver, self.config)
		if self.mpi and self.mpi.rank():
			self.bracket.head = 0
		else:
			self.bracket.head = 1

		self.lbfgs_s = []
		self.lbfgs_y = []
		self.lbfgs_mem = int(self.config['lbfgs_mem'])

	def restart_search(self):
		super().restart_search()
		self.lbfgs_s = []
		self.lbfgs_y = []

	def compute_direction(self):
		g_new = self.g_new

		if not hasattr(self, 'g_old'):
			return -g_new

		self.lbfgs_s.insert(0, self.m_new - self.m_old)
		self.lbfgs_y.insert(0, self.g_new - self.g_old)

		if len(self.lbfgs_s) > self.lbfgs_mem:
			self.lbfgs_s.pop()
			self.lbfgs_y.pop()

		nused = len(self.lbfgs_s)

		rh = np.zeros(nused)
		al = np.zeros(nused)

		q = np.copy(g_new)

		for i in range(nused):
			rh[i] = 1 / np.dot(self.lbfgs_y[i], self.lbfgs_s[i])
			al[i] = rh[i] * np.dot(self.lbfgs_s[i], q)
			q -= al[i] * self.lbfgs_y[i]

		q *= np.dot(self.lbfgs_y[0], self.lbfgs_s[0]) / np.dot(self.lbfgs_y[0], self.lbfgs_y[0])

		for i in range(nused-1, -1, -1):
			be = rh[i] * np.dot(self.lbfgs_y[i], q)
			q += self.lbfgs_s[i] * (al[i] - be)

		if self.bracket.angle(g_new, q) >= np.pi / 2:
			print('  restarting LBFGS: not descent')
			self.restart_search()
			return -g_new

		return -q

		

			
