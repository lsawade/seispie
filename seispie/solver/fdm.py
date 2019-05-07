from sys import modules
from os import path

import numpy as np

from seispie.solver.base import base

class fdm(base):
	def setup(self, config):
		if 'sh' in config:
			self.sh = config['sh']
		else:
			self.sh = 'no'

		if 'psv' in config:
			self.psv = config['psv']
		else:
			self.psv = 'no'

		if 'save_velocity' in config:
			self.save_velocity = config['save_velocity']
		else:
			self.save_velocity = 'no'

		if 'save_displacement' in config:
			self.save_displacement = config['save_displacement']
		else:
			self.save_displacement = 'no'

		if 'save_snapshot' in config:
			self.save_snapshot = config['save_snapshot']
		else:
			self.save_snapshot = 0

		if 'combine_sources' in config:
			self.combine_sources = config['combine_sources']
		else:
			self.combine_sources = 'no'

		if 'interpolate_model' in config:
			self.interpolate_model = config['interpolate_model']
		else:
			self.interpolate_model = 0

		assert self.sh == 'yes' or self.psv == 'yes'
		assert 'nt' in config
		assert 'dt' in config
		
		self.nt = config['nt']
		self.dt = config['dt']
		
	def import_model(self, dir):
		""" import model
		"""

		self.model = dict()
		
		for name in ['x', 'z', 'vp', 'vs', 'rho']:
			filename = path.join(dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				f.seek(0)
				self.npt = np.fromfile(f, dtype='int32', count=1)[0]

				f.seek(4)
				self.model[name] = np.fromfile(f, dtype='float32')

		x = self.model['x']
		z = self.model['z']
		lx = x.max() - x.min()
		lz = z.max() - z.min()
		self.nx = int(np.rint(np.sqrt(self.npt * lx / lz)))
		self.nz = int(np.rint(np.sqrt(self.npt * lz / lx)))
		self.dx = lx / (self.nx - 1)
		self.dz = lz / (self.nz - 1)

		for i in range(self.nx):
			for j in range(self.nz):
				idx = i*self.nz + j
				if (x[idx]-i*self.dx) != 0:
					print(i,j)

				if (z[idx]-j*self.dz) != 0:
					print(i,j)


	def run_forward(self):
		pass
		

