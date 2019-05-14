from seispie.optimize.cg import cg

class gd(cg):
	def compute_direction(self):
		return -self.g_new