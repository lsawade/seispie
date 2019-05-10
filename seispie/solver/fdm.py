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
def set_bound(bound, width, alpha, left, right, bottom, top, nx, nz):
	k, i, j = idxij(nz)
	if k < bound.size:
		bound[k] = 1;

		if left and i + 1 < width:
			aw = alpha * (width - i - 1)
			bound[k] *= math.exp(-aw * aw)


		if right and i > nx - width:
			aw = alpha * (width + i - nx)
			bound[k] *= math.exp(-aw * aw)

		if bottom and j > nz - width:
			aw = alpha * (width + j - nz)
			bound[k] *= math.exp(-aw * aw)

		if top and j + 1 < width:
			aw = alpha * (width - j - 1)
			bound[k] *= math.exp(-aw * aw)

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
	if k < dsx.size:
		if i >= 2 and i < nx - 2:
			dsx[k] = 9 * (sxx[k] - sxx[k-nz]) / (8 * dx) - (sxx[k+nz] - sxx[k-2*nz]) / (24 * dx)
			dsz[k] = 9 * (sxz[k] - sxz[k-nz]) / (8 * dx) - (sxz[k+nz] - sxz[k-2*nz]) / (24 * dx)
		else:
			dsx[k] = 0
			dsz[k] = 0

		if j >= 2 and j < nz - 2:
			dsx[k] += 9 * (sxz[k] - sxz[k-1]) / (8 * dz) - (sxz[k+1] - sxz[k-2]) / (24 * dz)
			dsz[k] += 9 * (szz[k] - szz[k-1]) / (8 * dz) - (szz[k+1] - szz[k-2]) / (24 * dz)

@cuda.jit
def add_dsy(dsy, stf_y, src_x, src_z, isrc, it, nt, nz):
	ib = cuda.blockIdx.x
	xs = src_x[ib]
	zs = src_z[ib]

	if isrc < 0 or isrc == ib:
		ks = ib * nt + it
		km = xs * nz + zs
		dsy[km] += stf_y[ks]

@cuda.jit
def add_dsxz(dsx, dsz, stf_x, stf_z, src_x, src_z, isrc, it, nt, nz):
	ib = cuda.blockIdx.x
	xs = src_x[ib]
	zs = src_z[ib]

	if isrc < 0 or isrc == ib:
		ks = ib * nt + it
		km = xs * nz + zs
		dsx[km] += stf_x[ks]
		dsz[km] += stf_z[ks]


class fdm(base):
	def setup(self):
		# FIXME validate config
		self.stream = cuda.stream()
		self.nt = int(self.config['nt'])
		self.dt = float(self.config['dt'])

	def import_sources(self, dir):
		src = np.loadtxt(dir, ndmin=2)
		nsrc = self.nsrc = src.shape[0]

		src_x = np.zeros(nsrc, dtype='int32')
		src_z = np.zeros(nsrc, dtype='int32')

		stf_x = np.zeros(nsrc * self.nt)
		stf_y = np.zeros(nsrc * self.nt)
		stf_z = np.zeros(nsrc * self.nt)

		for isrc in range(nsrc):
			src_x[isrc] = int(np.round(src[isrc][0] / self.dx))
			src_z[isrc] = int(np.round(src[isrc][1] / self.dz))

			for it in range(0, self.nt):
				istf = isrc * self.nt + it
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

		rec_x = np.zeros(nrec, dtype='int32')
		rec_z = np.zeros(nrec, dtype='int32')

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

		# allocate array
		stream = self.stream
		zeros = np.zeros(npt)

		self.lam = cuda.to_device(model['vp'], stream=stream)
		self.mu = cuda.to_device(model['vs'], stream=stream)
		self.rho = cuda.to_device(model['rho'], stream=stream)
		self.bound = cuda.to_device(zeros, stream=stream) # absorbing boundary

		abs_left = 1 if self.config['abs_left'] == 'yes' else 0
		abs_right = 1 if self.config['abs_right'] == 'yes' else 0
		abs_top = 1 if self.config['abs_top'] == 'yes' else 0
		abs_bottom = 1 if self.config['abs_bottom'] == 'yes' else 0

		set_bound[self.dim](
			self.bound, int(self.config['abs_width']), float(self.config['abs_alpha']),
			abs_left, abs_right, abs_bottom, abs_top, nx, nz
		);
		
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
		# self.bound.copy_to_host(t, stream=stream)
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
		sxz = self.sxz
		sxy = self.sxy
		szy = self.szy

		src_x = self.src_x
		src_z = self.src_z
		rec_x = self.rec_x
		rec_z = self.rec_z

		stf_x = self.stf_x
		stf_y = self.stf_y
		stf_z = self.stf_z

		bound = self.bound

		sh = self.config['sh']
		psv = self.config['psv']

		nsrc = self.nsrc
		nrec = self.nrec

		if self.config['combine_sources']:
			isrc = -1
		else:
			isrc = self.taskid

		for it in range(1):
		# for it in range(self.nt):
			# FIXME isfe
			# FIXME src_z, src_z => src
			if sh:
				div_sy[dim](dsy, sxy, szy, dx, dz, nx, nz)
				add_dsy[nsrc, 1](dsy, stf_y, src_x, src_z, isrc, it, nt, nz)
				# add_vy[dim](vy, uy, dsy, rho, absbound, dt, dim);
				# divVY<<<dim.dg, dim.db>>>(dvydx, dvydz, vy, dx, dz, dim);
				# updateSY<<<dim.dg, dim.db>>>(sxy, szy, dvydx, dvydz, mu, dt, dim);

			if psv:
				div_sxz[dim](dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz)
				add_dsxz[nsrc, 1](dsx, dsz, stf_x, stf_z, src_x, src_z, isrc, it, nt, nz)
				# updateVXZ<<<dim.dg, dim.db>>>(vx, vz, ux, uz, dsx, dsz, rho, absbound, dt, dim);
				# divVXZ<<<dim.dg, dim.db>>>(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, dim);
				# updateSXZ<<<dim.dg, dim.db>>>(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lambda, mu, dt, dim);

			print('id', self.taskid, isrc)
