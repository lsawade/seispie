from mpi4py import MPI

class mpi:
	def rank(self):
		return MPI.COMM_WORLD.Get_rank()
	
	def size(self):
		return MPI.COMM_WORLD.Get_size()

	def sync(self):
		MPI.COMM_WORLD.Barrier()

	def sum(self, data, out):
		MPI.COMM_WORLD.Allreduce(data, out, op=MPI.SUM)