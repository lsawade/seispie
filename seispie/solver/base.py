from os import path
import numpy as np

class base:
	def setup(self, config):
		raise NotImplementedError

	def runForward(self):
		raise NotImplementedError

	def runAdjoint(self):
		raise NotImplementedError

	def importModel(self, dir):
		names = ['vp', 'vs', 'rho', 'x', 'z']

		for name in names:
			filename = path.join(dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				f.seek(0)
				n = np.fromfile(f, dtype='int32', count=1)[0]
				self.n = n

				print(n, path.getsize(filename) )

				f.seek(4)
				data = np.fromfile(f, dtype='float32')

				self.model[name] = data[:-1]