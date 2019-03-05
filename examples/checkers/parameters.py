# general
system = 'serial'
workflow = 'inversion'
solver = 'specfem2d'
preprocess = 'default'
postprocess = 'default'
optimize = 'lbfgs'

# path
output = '$(pwd)/output'
model_init = '$(pwd)/model_init'
model_true = '$(pwd)/model_true'
# trace = '$(pwd)/trace'

# solver
source = 'ricker'
nsrc = 25
nrec = 132
channels = ['y']

# workflow
parameters = ['vs']
niter = 5
save_gradient = 1
save_misfit = 1

# postprocess
smooth = 20
ratio = 0.98
scale = 6.0e6

# optimize
misfit = 'waveform'
precond = 0
step_max = 10
step_thresh = 0.1