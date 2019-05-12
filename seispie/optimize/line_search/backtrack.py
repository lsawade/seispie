import numpy as np
from seispie.optimize.line_search.bracket import bracket

class backtrack(bracket):
	def calculate_step(self, step_count, step_max):
		x, f, update_count = self.get_history(step_count)

		if update_count==0:
			alpha, status = super().calculate_step(step_count, step_max)

		elif step_count==0:
			alpha = min(1., step_max)
			status = 0

		elif f.min() < f[0]:
			alpha = x[f.argmin()]
			status = 1

		elif step_count <= step_max:
			slope = self.ls_gtp[-1]/self.ls_gtg[-1]
			alpha = self.backtrack(f[0], slope, x[1], f[1], 0.1, 0.5)
			status = 0

		else:
			alpha = 0
			status = -1

		return alpha, status