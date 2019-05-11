from os import path
import numpy as np

class base:
	def setup(self, workflow):
		raise NotImplementedError

	def run_forward(self):
		raise NotImplementedError

	def run_adjoint(self):
		raise NotImplementedError

	def import_model(self, dir):
		raise NotImplementedError

		