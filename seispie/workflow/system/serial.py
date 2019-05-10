def call(self, ntask, target, method, *args):
	""" call module
	ntask = 0: 1 task per node
	ntask > 0: distribute n tasks to all nodes
	"""
	module = getattr(self, target)
	if ntask == 0:
		ntask = 1
		
	for i in range(ntask):
		module.taskid = i
		task = getattr(module, method)
		task(*args)