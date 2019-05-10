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
	k = tx + ty * bw
	return k

@cuda.jit(device=True)
def idxij(nz):
	k = idx()
	j = k % nz
	i = int((k - j) / nz)
	return k, i, j

@cuda.jit
def vps2lm(lam, mu, rho):
	k = idx()
	if k < lam.size:
		vp = lam[k]
		vs = mu[k]

		if vp > vs:
			lam[k] = rho[k] * (vp * vp - 2 * vs * vs)
		else:
			lam[k] = 0

		mu[k] = rho[k] * vs * vs

@cuda.jit
def lm2vps(vp, vs, rho):
	k = idx()
	if k < vp.size:
		lam = vp[k]
		mu = vs[k]

		vp[k] = math.sqrt((lam + 2 * mu) / rho[k])
		vs[k] = math.sqrt(mu / rho[k])

@cuda.jit
def div_sy(mu, nz):
	k, i, j = idxij(nz)
	mu[k] = i + j

@cuda.jit
def div_sy(dsy, sxy, szy, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < dsy.size:
		if i >= 2 and i < nx - 2:
			dsy[k] = 9 * (sxy[k] - sxy[k-nz]) / (8 * dx) - (sxy[k+nz] - sxy[k-2*nz]) / (24 * dx)
		else:
			dsy[k] = 0

		if j >= 2 and j < nz - 2:
			dsy[k] += 9 * (szy[k] - szy[k-1]) / (8 * dz) - (szy[k+1] - szy[k-2]) / (24 * dz)

@cuda.jit
def div_sxz(dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < dsy.size:
		if i >= 2 and i < nx - 2:
			dsx[k] = 9 * (sxx[k] - sxx[k-nz]) / (8 * dx) - (sxx[k+nz] - sxx[dim(k-2*nz)]) / (24 * dx);
			dsz[k] = 9 * (sxz[k] - sxz[k-nz]) / (8 * dx) - (sxz[k+nz] - sxz[dim(k-2*nz)]) / (24 * dx);
		else:
			dsx[k] = 0;
			dsz[k] = 0;
		
		if j >= 2 and j < nz - 2:
			dsx[k] += 9 * (sxz[k] - sxz[k-1]) / (8 * dz) - (sxz[k+1] - sxz[k-2]) / (24 * dz)
			dsz[k] += 9 * (szz[k] - szz[k-1]) / (8 * dz) - (szz[k+1] - szz[k-2]) / (24 * dz)

class fdm(base):
	def setup(self):
		# FIXME validate config
		self.stream = cuda.stream()
		self.nt = int(self.config['nt'])
		self.dt = float(self.config['dt'])

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
		
		# allocate array
		stream = self.stream

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
				if not hasattr(self, 'npt'):
					f.seek(0)
					self.npt = np.fromfile(f, dtype='int32', count=1)

				f.seek(4)
				model[name] = np.fromfile(f, dtype='float32').astype('float64')

		npt = self.npt[0]
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


		for i in range(nx):
			for j in range(nz):
				pos = i * nz + j
				if x[pos] != i * dx:
					print(x[pos], i*dx)
				
				if z[pos] != j * dz:
					print(z[pos], j*dz)

		print('verified')

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

		# # FIXME remove below
		# lm2vps[self.dim](self.lam, self.mu, self.rho)
		# t = np.zeros(npt)
		# div_sy[self.dim](self.mu, self.nz)
		# self.mu.copy_to_host(t, stream=stream)
		# stream.synchronize()
		# self.export_field(t, 'vq')
		# print('valalaah', t[0])

	def export_field(self, field, name):
		with open(self.path['output'] + '/proc000000_' + name + '.bin', 'w') as f:
			f.seek(0)
			self.npt.tofile(f)

			f.seek(4)
			field.astype('float32').tofile(f)


	def run_forward(self):
		stream = self.stream
		dim = self.dim

		nx = self.nx
		nz = self.nz
		nt = self.nt
		dx = self.dx
		dz = self.dz
		dt = self.dt

		dsx = self.dsx
		dsy = self.dsy
		dsz = self.dsz
		sxx = self.sxx
		szz = self.szz
		sxy = self.sxy
		szy = self.szy


		for it in range(1):
		# for it in range(self.nt):
			# FIXME isfe
			if self.config['sh']:
				div_sy[dim](dsy, sxy, szy, dx, dz, nx, nz);

			if self.config['psv']:
				div_sxz[dim](dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz);