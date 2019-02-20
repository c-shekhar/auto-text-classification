from imports import pd,np

class Read(object):
	def __init__(self):
		self.train_df = None
		self.test_df = None

	def read_from_csv(self, train_path, test_path):
		train = pd.read_csv(train_path,header=0)
		test = pd.read_csv(test_path,header=0)
		self.train_df = train
		self.test_df = test
		return train, test


class Write(object):
	def __init__(self):
		pass

	def to_csv(self):
		pass
