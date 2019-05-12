import numpy as np

class bracket:
	def __init__(self, solver, config):
		self.solver = solver

		self.ls_lens = []
		self.ls_vals = []
		self.ls_gtg = []
		self.ls_gtp = []

		self.ls_step = int(config['ls_step'])
		self.ls_step_max = float(config['ls_step_max'])
		self.ls_step_init = float(config['ls_step_init'])
		self.ls_thresh = float(config['ls_thresh'])

	def restart(self):
		self.ls_lens = []
		self.ls_vals = []
		self.ls_gtg = []
		self.ls_gtp = []

	def angle(self, x, y):
		xy = np.dot(x,y)
		xx = np.dot(x,x)
		yy = np.dot(y,y)
		ang = xy/(xx*yy)**0.5
		if abs(ang) > 1:
			ang /= abs(ang)
		return np.arccos(ang)

	def run(self, m_new, g_new, p_new, f):
		if self.head:
			print('  step 1')

		solver = self.solver
		status = 0
		alpha = 0

		norm_m = np.amax(abs(m_new))
		norm_p = np.amax(abs(p_new))
		gtg = np.dot(g_new, g_new)
		gtp = np.dot(g_new, p_new)

		step_max = self.ls_step_max * norm_m / norm_p
		step_count = 0
		self.ls_lens.append(0)
		self.ls_vals.append(f)
		self.ls_gtg.append(gtg)
		self.ls_gtp.append(gtp)

		if self.ls_step_init and len(self.ls_lens) <= 1:
			alpha = self.ls_step_init * norm_m / norm_p
		else:
			alpha, status = self.calculate_step(step_count, step_max)
		

		while(True):
			solver.update_model(m_new + alpha * p_new)
			self.ls_lens.append(alpha)
			self.ls_vals.append(solver.compute_misfit())
			step_count += 1

			alpha, status = self.calculate_step(step_count, step_max)
			if self.head:
				print('  step %d' % (step_count+1))

			if status > 0:
				solver.update_model(m_new + alpha * p_new)
				return status
			
			elif status < 0:
				solver.update_model(m_new)
				if self.angle(p_new, -g_new) < 1e-3:
					return status
				else:
					if self.head:
						print('  restarting line search')
						
					self.restart()
					return self.run(m_new, g_new, p_new, f)

	def get_history(self, i):
		k = len(self.ls_lens)
		x = np.array(self.ls_lens[k-i-1:k])
		f = np.array(self.ls_vals[k-i-1:k])
		f = f[abs(x).argsort()]
		x = x[abs(x).argsort()]
		return x, f, sum(np.array(self.ls_lens) == 0)-1

	def calculate_step(self, step_count, step_max):
		x, f, update_count = self.get_history(step_count)

		if step_count == 0:
			if update_count == 0:
				alpha = 1 / self.ls_gtg[-1]
				status = 0
			else:
				idx = np.argmin(self.ls_vals[:-1])
				alpha = self.ls_lens[idx] * self.ls_gtp[-2] / self.ls_gtp[-1]
				status = 0
				
		elif self.check_bracket(x, f):
			if self.good_enough(x, f):
				alpha = x[f.argmin()]
				status = 1
			else:
				alpha = self.polyfit(x,f)
				status = 0

		elif step_count <= self.ls_step:
			if all(f <= f[0]):
				alpha = 1.618034 * x[step_count]
				status = 0
			else:
				slope = self.ls_gtp[-1]/self.ls_gtg[-1]
				alpha = self.backtrack(f[0], slope, x[1], f[1], 0.1, 0.5)
				status = 0

		else:
			alpha = 0
			status = -1

		if alpha > step_max:
			if step_count == 0:
				alpha = 0.618034 * step_max
				status = 0
			else:
				alpha = step_max
				status = 1

		return alpha, status

	def check_bracket(self, x, f):
		imin, fmin = f.argmin(), f.min()
		if (fmin < f[0]) and any(f[imin:] > fmin):
			return 1
		else:
			return 0

	def good_enough(self, x, f):
		if not self.check_bracket(x,f):
			return 0
		x0 = self.polyfit(x,f)
		if any(np.abs(np.log10(x[1:]/x0)) < np.log10(1.2)):
			return 1
		else:
			return 0

	def polyfit(self, x, f):
		i = np.argmin(f)
		p = np.polyfit(x[i-1:i+2], f[i-1:i+2], 2)
		if p[0] > 0:
			return -p[1]/(2*p[0])
		else:
			raise Exception()

	def backtrack(self, f0, g0, x1, f1, b1, b2):
		x2 = -g0*x1**2/(2*(f1-f0-g0*x1))
		if x2 > b2*x1:
			x2 = b2*x1
		elif x2 < b1*x1:
			x2 = b1*x1
		return x2

	