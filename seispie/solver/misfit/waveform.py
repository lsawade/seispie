from numba import cuda
import math
import numpy as np

@cuda.jit
def diff(syn, obs, adstf, misfit, nt, dt):
	it = cuda.blockIdx.x
	ir = cuda.threadIdx.x
	kt = ir * nt + it
	akt = ir * nt + nt - it - 1

	t_end = (nt - 1) * dt
	taper_width = t_end / 10
	t_min = taper_width
	t_max = t_end - taper_width
	t = it * dt
	if t <= t_min:
		taper = 0.5 + 0.5 * math.cos(math.pi * (t_min - t) / taper_width)
	elif t >= t_max:
		taper = 0.5 + 0.5 * math.cos(math.pi * (t_max - t) / taper_width)
	else:
		taper = 1

	misfit[kt] = (syn[kt] - obs[kt]) * taper
	adstf[akt] = misfit[kt] * taper * 2


def waveform(syn, obs, adstf, nt, dt, nrec, stream):
	misfit = np.zeros(nt * nrec, dtype='float32')
	d_misfit = cuda.to_device(misfit, stream=stream)
	diff[nt, nrec](syn, obs, adstf, d_misfit, nt, dt)
	d_misfit.copy_to_host(misfit, stream=stream)
	stream.synchronize()
	return np.linalg.norm(misfit)