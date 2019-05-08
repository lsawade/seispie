import numpy as np
import math

from os import path
from numba import cuda

from seispie.solver.base import base
from seispie.solver.source.ricker import ricker

@cuda.jit(device=True)
def idx():
	tx = cuda.threadIdx.x
	ty = cuda.blockIdx.x
	bw = cuda.blockDim.x
	pos = tx + ty * bw
	return pos

@cuda.jit
def vps2lm(lam, mu, rho):
	pos = idx()
	if pos < lam.size:
		vp = lam[pos]
		vs = mu[pos]

		if vp > vs:
			lam[pos] = rho[pos] * (vp * vp - 2 * vs * vs)
		else:
			lam[pos] = 0

		mu[pos] = rho[pos] * vs * vs

@cuda.jit
def lm2vps(vp, vs, rho):
	pos = idx()
	if pos < vp.size:
		lam = vp[pos]
		mu = vs[pos]

		vp[pos] = math.sqrt((lam + 2 * mu) / rho[pos])
		vs[pos] = math.sqrt(mu / rho[pos])

class fdm(base):
	def setup(self, config):
		# FIXME validate config
		self.config = config
		self.stream = cuda.stream()

		self.nt = int(config['nt'])
		self.dt = float(config['dt'])

	def import_sources(self, dir):
		src = np.loadtxt(dir)
		nsrc = self.nsrc = src.shape[0]

		src_x = np.zeros(nsrc)
		src_z = np.zeros(nsrc)

		stf_x = np.zeros(nsrc * self.nt)
		stf_y = np.zeros(nsrc * self.nt)
		stf_z = np.zeros(nsrc * self.nt)

		for isrc in range(nsrc):
			src_x[isrc] = int(np.round(src[isrc][0] / self.dx))
			src_z[isrc] = int(np.round(src[isrc][1] / self.dz))

			for it in range(0, self.nt):
				istf = isrc * self.nt + it

				# FIXME source time function
				stf_x[istf], stf_y[istf], stf_z[istf] = ricker(it * self.dt, *src[isrc][3:])

		# allocate array
		stream = self.stream

		self.src_x = cuda.to_device(src_x, stream=stream)
		self.src_z = cuda.to_device(src_z, stream=stream)

		self.stf_x = cuda.to_device(stf_x, stream=stream)
		self.stf_y = cuda.to_device(stf_y, stream=stream)
		self.stf_z = cuda.to_device(stf_z, stream=stream)

	def import_stations(self, dir):
		rec = np.loadtxt(dir)
		nrec = self.nrec = rec.shape[0]

		rec_x = np.zeros(nrec)
		rec_z = np.zeros(nrec)

		for irec in range(nrec):
			rec_x[irec] = int(np.round(rec[irec][0] / self.dx))
			rec_z[irec] = int(np.round(rec[irec][1] / self.dz))

		self.rec_x = cuda.to_device(rec_x, stream=stream)
		self.rec_z = cuda.to_device(rec_z, stream=stream)
		
	def import_model(self, dir):
		""" import model
		"""
		model = self.imported_model = dict()
		
		for name in ['x', 'z', 'vp', 'vs', 'rho']:
			filename = path.join(dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				f.seek(0)
				npt = np.fromfile(f, dtype='int32', count=1)[0]

				f.seek(4)
				model[name] = np.fromfile(f, dtype='float32').astype('float64')

		ntpb = int(self.config['threads_per_block'])
		nb = int(np.ceil(npt / ntpb))
		self.dim = nb, ntpb

		x = model['x']
		z = model['z']
		lx = x.max() - x.min()
		lz = z.max() - z.min()
		nx = self.nx = int(np.rint(np.sqrt(npt * lx / lz)))
		nz = self.nz = int(np.rint(np.sqrt(npt * lz / lx)))
		dx = self.dx = lx / (nx - 1)
		dz = self.dz = lz / (nz - 1)

		# allocate array
		stream = self.stream
		zeros = np.zeros(npt)

		self.lam = cuda.to_device(model['vp'], stream=stream)
		self.mu = cuda.to_device(model['vs'], stream=stream)
		self.rho = cuda.to_device(model['rho'], stream=stream)
		self.abs = cuda.to_device(zeros, stream=stream) # absorbing boundary
		
		if self.config['sh'] == 'yes':
			self.vy = cuda.to_device(zeros, stream=stream)
			self.uy = cuda.to_device(zeros, stream=stream)
			self.sxy = cuda.to_device(zeros, stream=stream)
			self.szy = cuda.to_device(zeros, stream=stream)
			self.dsy = cuda.to_device(zeros, stream=stream)
			self.dvydx = cuda.to_device(zeros, stream=stream)
			self.dvydz = cuda.to_device(zeros, stream=stream)

		if self.config['psv'] == 'yes':
			self.vx = cuda.to_device(zeros, stream=stream)
			self.vz = cuda.to_device(zeros, stream=stream)
			self.ux = cuda.to_device(zeros, stream=stream)
			self.uz = cuda.to_device(zeros, stream=stream)
			self.sxx = cuda.to_device(zeros, stream=stream)
			self.szz = cuda.to_device(zeros, stream=stream)
			self.sxz = cuda.to_device(zeros, stream=stream)
			self.dsx = cuda.to_device(zeros, stream=stream)
			self.dsz = cuda.to_device(zeros, stream=stream)
			self.dvxdx = cuda.to_device(zeros, stream=stream)
			self.dvxdz = cuda.to_device(zeros, stream=stream)
			self.dvzdx = cuda.to_device(zeros, stream=stream)
			self.dvzdz = cuda.to_device(zeros, stream=stream)

		# FIXME interpolate model

		# change parameterization
		vps2lm[self.dim](self.lam, self.mu, self.rho)

		# FIXME remove below
		# lm2vps[self.dim](self.lam, self.mu, self.rho)
		# t = np.zeros(npt)
		# self.mu.copy_to_host(t, stream=stream)
		# stream.synchronize()
		# print('valalaah', t[0])

	def run_forward(self):
		pass
		# stream = self.stream

		# div_stress[nsrc, 1]
		

