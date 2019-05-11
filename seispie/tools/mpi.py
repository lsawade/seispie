from mpi4py import MPI

class mpi:
	def rank(self):
		return MPI.COMM_WORLD.Get_rank()
	
	def size(self):
		return MPI.COMM_WORLD.Get_size()