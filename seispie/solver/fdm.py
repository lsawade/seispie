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

@cuda.jit('void(float32[:], float32[:], float32[:], float32, float32, int32, int32)')
def div_sy(dsy, sxy, szy, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < nx * nz:
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

@cuda.jit('void(float32[:], float32[:], int32[:], int32, int32, int32)')
def stf_dsy(dsy, stf_y, src_id, isrc, it, nt):
	ib = cuda.blockIdx.x
	if isrc < 0 or isrc == ib:
		ks = ib * nt + it
		km = src_id[ib]
		dsy[km] += stf_y[ks]

@cuda.jit
def stf_dsxz(dsx, dsz, stf_x, stf_z, src_id, isrc, it, nt):
	ib = cuda.blockIdx.x
	if isrc < 0 or isrc == ib:
		ks = ib * nt + it
		km = src_id[ib]
		dsx[km] += stf_x[ks]
		dsz[km] += stf_z[ks]

@cuda.jit('void(float32[:], float32[:], float32[:], float32[:], float32[:], float32, int32)')
def add_vy(vy, uy, dsy, rho, bound, dt, npt):
	k = idx()
	if k < npt:
		vy[k] = bound[k] * (vy[k] + dt * dsy[k] / rho[k])
		uy[k] += vy[k] * dt

@cuda.jit
def add_vxz(vx, vz, ux, uz, dsx, dsz, rho, bound, dt):
	k = idx()
	if k < vx.size:
		vx[k] = bound[k] * (vx[k] + dt * dsx[k] / rho[k])
		vz[k] = bound[k] * (vz[k] + dt * dsz[k] / rho[k])
		ux[k] += vx[k] * dt;
		uz[k] += vz[k] * dt;

@cuda.jit('void(float32[:], float32[:], float32[:], float32, float32, int32, int32)')
def div_vy(dvydx, dvydz, vy, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < nx * nz:
		if i >= 1 and i < nx - 2:
			dvydx[k] = 9 * (vy[k+nz] - vy[k]) / (8 * dx) - (vy[k+2*nz] - vy[k-nz]) / (24 * dx)
		else:
			dvydx[k] = 0
		if j >= 1 and j < nz - 2:
			dvydz[k] = 9 * (vy[k+1] - vy[k]) / (8 * dz) - (vy[k+2] - vy[k-1]) / (24 * dz)
		else:
			dvydz[k] = 0

@cuda.jit
def div_vxz(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, nx, nz):
	k, i, j = idxij(nz)
	if k < dvxdx.size:
		if i >= 1 and i < nx - 2:
			dvxdx[k] = 9 * (vx[k+nz] - vx[k]) / (8 * dx) - (vx[k+2*nz] - vx[k-nz]) / (24 * dx)
			dvzdx[k] = 9 * (vz[k+nz] - vz[k]) / (8 * dx) - (vz[k+2*nz] - vz[k-nz]) / (24 * dx)
		else:
			dvxdx[k] = 0
			dvzdx[k] = 0		
		if j >= 1 and j < nz - 2:
			dvxdz[k] = 9 * (vx[k+1] - vx[k]) / (8 * dz) - (vx[k+2] - vx[k-1]) / (24 * dz)
			dvzdz[k] = 9 * (vz[k+1] - vz[k]) / (8 * dz) - (vz[k+2] - vz[k-1]) / (24 * dz)
		else:
			dvxdz[k] = 0
			dvzdz[k] = 0	

@cuda.jit('void(float32[:], float32[:], float32[:], float32[:], float32[:], float32, int32)')
def add_sy(sxy, szy, dvydx, dvydz, mu, dt, npt):
	k = idx()
	if k < npt:
		sxy[k] += dt * mu[k] * dvydx[k]
		szy[k] += dt * mu[k] * dvydz[k]

@cuda.jit
def add_sxz(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lam, mu, dt):
	k = idx()
	if k < sxx.size:
		sxx[k] += dt * ((lam[k] + 2 * mu[k]) * dvxdx[k] + lam[k] * dvzdz[k])
		szz[k] += dt * ((lam[k] + 2 * mu[k]) * dvzdz[k] + lam[k] * dvxdx[k])
		sxz[k] += dt * (mu[k] * (dvxdz[k] + dvzdx[k]))

class fdm(base):
	def setup(self):
		# FIXME validate config
		self.stream = cuda.stream()
		self.nt = int(self.config['nt'])
		self.dt = float(self.config['dt'])

	def import_sources(self):
		src = np.loadtxt(self.path['sources'], ndmin=2)
		nsrc = self.nsrc = src.shape[0]

		src_id = np.zeros(nsrc, dtype='int32')

		stf_x = np.zeros(nsrc * self.nt, dtype='float32')
		stf_y = np.zeros(nsrc * self.nt, dtype='float32')
		stf_z = np.zeros(nsrc * self.nt, dtype='float32')

		for isrc in range(nsrc):
			src_x = int(np.round(src[isrc][0] / self.dx))
			src_z = int(np.round(src[isrc][1] / self.dz))
			src_id[isrc] = src_x * self.nz + src_z

			for it in range(0, self.nt):
				istf = isrc * self.nt + it
				stf_x[istf], stf_y[istf], stf_z[istf] = ricker(it * self.dt, *src[isrc][3:])

		# allocate array
		stream = self.stream

		self.src_id = cuda.to_device(src_id, stream=stream)
		self.stf_x = cuda.to_device(stf_x, stream=stream)
		self.stf_y = cuda.to_device(stf_y, stream=stream)
		self.stf_z = cuda.to_device(stf_z, stream=stream)

	def import_stations(self):
		rec = np.loadtxt(self.path['stations'])
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

	def import_model(self, model_true):
		""" import model
		"""
		model = dict()
		model_dir = self.path['model_true'] if model_true else self.path['model_init']

		for name in ['x', 'z', 'vp', 'vs', 'rho']:
			filename = path.join(model_dir, 'proc000000_' + name + '.bin')
			with open(filename) as f:
				if not hasattr(self, 'npt'):
					f.seek(0)
					self.npt = np.fromfile(f, dtype='int32', count=1)

				f.seek(4)
				model[name] = np.fromfile(f, dtype='float32')

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
		zeros = np.zeros(npt, dtype='float32')

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
		
		dats = []

		if self.config['sh'] == 'yes':
			dats += ['vy', 'uy', 'sxy', 'szy', 'dsy', 'dvydx', 'dvydz']

		if self.config['psv'] == 'yes':
			dats += [
				'vx', 'vz', 'ux',' uz', 'sxx', 'szz', 'sxz',
				'dsx', 'dsz','dvxdx', 'dvxdz', 'dvzdx', 'dvzdz'
			]

		for dat in dats:
			setattr(self, dat, cuda.to_device(zeros, stream=stream))

		# FIXME interpolate model

		# change parameterization
		vps2lm[self.dim](self.lam, self.mu, self.rho)

		# write coordinate file
		if self.config['save_coordinates']:
			self.export_field(x, 'proc000000_x')
			self.export_field(z, 'proc000000_z')
		
		# # FIXME remove below
		# lm2vps[self.dim](self.lam, self.mu, self.rho)
		# t = np.zeros(npt)
		# self.bound.copy_to_host(t, stream=stream)
		# stream.synchronize()
		# self.export_field(t, 'vq')
		# print('valalaah', t[0])

	def export_field(self, field, name):
		with open(self.path['output'] + '/' + name + '.bin', 'w') as f:
			f.seek(0)
			self.npt.tofile(f)

			f.seek(4)
			field.tofile(f)


	def run_forward(self):
		stream = self.stream
		dim = self.dim
		sh = 1 if self.config['sh'] == 'yes' else 0
		psv = 1 if self.config['psv'] == 'yes' else 0

		nx = self.nx
		nz = self.nz
		nt = self.nt
		dx = self.dx
		dz = self.dz
		dt = self.dt

		if self.config['combine_sources'] == 'yes':
			isrc = -1
		else:
			isrc = self.taskid

		npt = self.npt
		out = np.zeros(npt, dtype='float32')
		sfe = int(self.config['save_snapshot'])

		for it in range(self.nt):
			# FIXME isfe
			if sh:
				div_sy[dim](self.dsy, self.sxy, self.szy, dx, dz, nx, nz)
				stf_dsy[self.nsrc, 1](self.dsy, self.stf_y, self.src_id, isrc, it, nt)
				add_vy[dim](self.vy, self.uy, self.dsy, self.rho, self.bound, dt, npt)
				div_vy[dim](self.dvydx, self.dvydz, self.vy, dx, dz, nx, nz)
				add_sy[dim](self.sxy, self.szy, self.dvydx, self.dvydz, self.mu, dt, npt)

			if psv:
				div_sxz[dim](self.dsx, self.dsz, self.sxx, self.szz, self.sxz, dx, dz, nx, nz)
				stf_dsxz[self.nsrc, 1](self.dsx, self.dsz, self.stf_x, self.stf_z, self.src_id, isrc, it, nt)
				add_vxz[dim](self.vx, self.vz, self.ux, self.uz, self.dsx, self.dsz, rho, bound, dt)
				div_vxz[dim](self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.vx, self.vz, dx, dz, nx, nz)
				add_sxz[dim](self.sxx, self.szz, self.sxz, self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.lam, self.mu, dt)

			if it > 0 and it % sfe == 0:
				if sh:
					self.vy.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'proc%06d_vy' % (it))

				if psv:
					self.vx.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'proc%06d_vx' % (it))
					self.vz.copy_to_host(out, stream=stream)
					stream.synchronize()
					self.export_field(out, 'proc%06d_vz' % (it))			


		print('id', self.taskid, isrc)
