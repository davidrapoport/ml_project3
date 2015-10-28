
# base classifier

class BaseClassifier(object):
	def fit(self):
		"""
			Fit the data to the classifier
		"""
		raise NotImplementedError()
	
	def predict(self):
		"""
			Predict class using a new observation
		"""
		raise NotImplementedError()

	def score(self):
		"""
			Score a new observation
		"""
		raise NotImplementedError()

	def predict_proba(self):
		"""
			Returns the probability of a certain class
		"""
		raise NotImplementedError()

