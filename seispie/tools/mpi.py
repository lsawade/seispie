from mpi4py import MPI as _MPI

class MPI:
	def call(self):
		pass

	def rank(self):
		return _MPI.COMM_WORLD.Get_rank()

	def size(self):
		return _MPI.COMM_WORLD.Get_size()
