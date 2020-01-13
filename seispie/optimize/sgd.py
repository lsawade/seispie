from seispie.optimize.gd import gd

class sgd(gd):
	def compute_direction(self):
		return -self.g_new