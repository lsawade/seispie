class base:
	""" workflow
	"""

	def setup(self, config):
		""" initialize
		"""
		raise NotImplementedError

	def run(self):
		""" start workflow
		"""
		raise NotImplementedError

	@property
	def modules(self):
		""" modules to load
		"""
		return []