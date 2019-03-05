from sys import modules
import numpy as np

from . base import base

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

		assert self.sh == 'yes' or self.psv == 'yes'
		assert 'nt' in config
		assert 'dt' in config
		
		self.nt = config['nt']
		self.dt = config['dt']
		self.model = dict()
		
	def importModel(self, dir):
		super().importModel(dir)
		print(self.model['z'][0:10])


	def runForward(self):
		pass
		

