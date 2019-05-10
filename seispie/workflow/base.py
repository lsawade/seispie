import sys

class base:
	""" workflow
	"""

	def setup(self):
		""" initialize
		"""
		raise NotImplementedError

	def run(self):
		""" start workflow
		"""
		raise NotImplementedError

	def call(self):
		raise NotImplementedError

	@property
	def modules(self):
		""" modules to load
		"""
		return []