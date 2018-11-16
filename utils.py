class RunningAverage():
	def __init__(self):
		self.steps = 0
		self.total = 0

	def update(self, val):
		self.update_step(val, 1)

	def update_step(self, val, step):
		self.total += (val * step)
		self.steps += step

	def __call__(self):
		return self.total/float(self.steps)