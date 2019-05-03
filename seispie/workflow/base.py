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
		from mpi4py import MPI

		size = MPI.COMM_WORLD.Get_size()
		rank = MPI.COMM_WORLD.Get_rank()
		name = MPI.Get_processor_name()

		msg = "Hello World! I am process {0} of {1} on {2}.\n"
		sys.stdout.write(msg.format(rank, size, name))

	@property
	def modules(self):
		""" modules to load
		"""
		return []