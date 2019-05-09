from mpi4py import MPI
import sys

def call(self):
	""" call module
	"""

	size = MPI.COMM_WORLD.Get_size()
	rank = MPI.COMM_WORLD.Get_rank()
	name = MPI.Get_processor_name()

	msg = "Hello World! I am process {0} of {1} on {2}.\n"
	sys.stdout.write(msg.format(rank, size, name))