def call(self, ntask, target, method, *args):
	""" call module
	ntask = 0: 1 task per node
	ntask > 0: distribute n tasks to all nodes
	"""
	module = getattr(self, target)
	task = getattr(module, method)
	task(*args)