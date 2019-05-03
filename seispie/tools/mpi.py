from mpi4py import MPI

class MPI:
	def call(self):
		pass

	def rank(self):
		return MPI.COMM_WORLD.Get_rank()

	def size(self):
		return MPI.COMM_WORLD.Get_size()