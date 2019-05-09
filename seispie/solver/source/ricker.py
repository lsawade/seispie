import numpy as np

def ricker(t, f0, t0, ang, amp):
	stf = -amp * (t - t0) * np.exp(-(np.pi * f0 * (t - t0)) ** 2)
	return stf * np.cos(ang), stf, stf * np.sin(ang)