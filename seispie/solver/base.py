from os import path
import numpy as np

class base:
	def setup(self, config):
		raise NotImplementedError

	def run_forward(self):
		raise NotImplementedError

	def run_adjoint(self):
		raise NotImplementedError

	def import_model(self, dir):
		names = ['vp', 'vs', 'rho', 'x', 'z']

		for name in names:
			filename = path.join(dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				f.seek(0)
				npt = np.fromfile(f, dtype='int32', count=1)[0]
				self.npt = npt

				f.seek(4)
				data = np.fromfile(f, dtype='float32')

				self.model[name] = data[:-1]

		x = self.model['x']
		z = self.model['z']
		lx = x.max() - x.min()
		lz = z.max() - z.min()
		self.nx = int(np.rint(np.sqrt(self.npt * lx / lz)))
		self.nz = int(np.rint(np.sqrt(self.npt * lz / lx)))
		self.dx = lx / (self.nx - 1)
		self.dz = lz / (self.nz - 1)

		