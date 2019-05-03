import sys

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

	def call(self):
		sys.stdout.write('Hello world')

	def call_mpi(self):
		if not hasattr(self, 'mpi'):
			from seispie.tools.mpi import MPI
			self.mpi = MPI()

		rank = self.mpi.rank()
		size = self.mpi.size()

		msg = "Hello World! I am process {0} of {1}.\n"
		sys.stdout.write(msg.format(rank, size))

	@property
	def modules(self):
		""" modules to load
		"""
		return []