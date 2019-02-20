from imports import RandomForestClassifier
from imports import LogisticRegression
from imports import confusion_matrix

class ClassificationModels(object):
	
	def __init__(self):
		self.model = None
		self.cm = None

	def create_logistic_regression(self):
		lr = LogisticRegression()
		self.model = lr
		return lr

	def create_random_forest(self):
		rf = RandomForestClassifier(n_estimators=100)
		self.model = rf
		return rf

	def train(self, X, y):
		return self.model.fit(X,y)

	def predict(self, X):
		return self.model.predict(X)

	def confusion_matrix(self, y_true, y_preds):
		cm = confusion_matrix(y_true,y_preds)
		self.cm = cm
		return cm

	def beautiful_cm(self, y_true, y_preds):
		cm = confusion_matrix(y_true,y_preds)
		self.cm = cm
		# beautify cm
		return cm
