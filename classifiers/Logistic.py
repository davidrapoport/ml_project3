import numpy as np
from util import gradient_descent, sigmoid


from sklearn.base import ClassifierMixin, BaseEstimator

# @TODO use sklearn baseclassifier
# sklearn.base 
class BinaryLogisticRegression( BaseEstimator, ClassifierMixin ):
	def __init__( self, learning_rate = 0.001, max_iterations = 100000, add_bias = True ):
		"""
			Creates a Binary Logistic Regression Classifier
			@params:
				learning_rate[0.001]: the maximum number of iterations to conduct gradient descent for
				max_iterations[100000]: the maximum number of iterations to conduct gradient descent for 
				add_bias[True]: adds a bias term to the data
		"""
		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.add_bias = add_bias
	
	def fit( self, X, Y ):
		"""
			Trains the model according to inputs Y
			Note this is a binary classifier.
			@params:
				X = shape: (n, m)
				Y = shape: (n, 1)
				Y must have only two unique classes
		"""
		n, m = X.shape
		assert n == Y.shape[0], 'Y must have same number of observations as X'
		assert Y.shape[1] == 1, 'Y must be a vector of dimension (n, 1)'

		# save the type of the output labels.
		self.output_type = Y.dtype

		if self.add_bias:
			# add the bias term
			bias_term = np.ones( ( n, 1 ) )
			X = np.append( bias_term, X, axis = 1 )

		self.labels = np.unique(Y) # get the uniqe labels in the dataset
		assert len(self.labels) == 2, 'BinaryLogisticRegression is a BINARY classifier. Y has more than two classes'

		Y_modified = Y.copy()

		# map our labels to 0, 1s
		Y_modified[ Y_modified == self.labels[0] ] = 0
		Y_modified[ Y_modified == self.labels[1] ] = 1
		Y_modified = Y_modified.astype(int)

		self.w_estimated = gradient_descent( X, Y_modified , alpha = self.learning_rate, max_iterations = self.max_iterations )

	def predict_proba( self, X ):
		"""
			Returns the probability of class being 1
			@params:
				X: observations to predict of second axis m
		"""
		if self.add_bias:
			# add the bias term
			bias_term = np.ones( ( X.shape[0], 1 ) )
			X = np.append( bias_term, X, axis = 1 )

		return sigmoid( np.dot( X, self.w_estimated ) )

	def predict( self, X ):
		"""
			Returns the predicted classes
		"""
		predictions = np.zeros( ( X.shape[0], 1 ) ).astype(self.output_type)

		probability_of_class_one = self.predict_proba(X)
		predictions[ probability_of_class_one >= 0.5 ] = self.labels[1]
		predictions[ probability_of_class_one < 0.5 ] = self.labels[0]

		return predictions

class LogisticRegression( BaseEstimator, ClassifierMixin ):
	def __init__( self, learning_rate = 0.001, max_iterations = 100000, add_bias = True ):
		"""
			MultiClass Logistic Regression Classifier.
			This classifier is implemented with the 1 vs all strategy.
			@params:
				learning_rate[0.001]: the maximum number of iterations to conduct gradient descent for
				max_iterations[100000]: the maximum number of iterations to conduct gradient descent for 
				add_bias[True]: adds a bias term to the data

		"""
		self.learning_rate = learning_rate
		self.add_bias = add_bias
		self.max_iterations = max_iterations

	def fit( self, X, Y ):
		"""
			Trains the model according to inputs Y
			Note this is a binary classifier.
			@params:
				X = shape: (n, m)
				Y = shape: (n, 1)
		"""

		n, m = X.shape
		assert n == Y.shape[0], 'Y must have same number of observations as X'
		Y = Y.reshape(n, 1)
		# assert Y.shape[1] == 1, 'Y must be a vector of dimension (n, 1)'

		# save the type of the output labels.
		self.output_type = Y.dtype

		self.labels = np.unique(Y) # get the uniqe labels in the dataset

		self._predictors = []

		# train individual predictors
		for label_index, label in enumerate( self.labels ):
			self._predictors.append( BinaryLogisticRegression( **self.get_params() ) )

			Y_modified = Y.copy()

			non_marker = hash(np.random.rand())
			# map our labels to 0, 1s
			Y_modified[ Y_modified != label ] = non_marker 
			Y_modified[ Y_modified == label ] = 1
			Y_modified[ Y_modified != 1 ] = 0
			
			Y_modified = Y_modified.astype(int)

			self._predictors[ label_index ].fit( X, Y_modified )


	def predict_proba( self, X ):
		"""
			Returns the tuple of probability of being in each class
		"""
		k = len(self.labels)
		n = X.shape[0]
		probability_predictions_matrix =  np.zeros( ( n, k ) )

		for label_index, label in enumerate( self.labels ):
			probability_predictions_matrix[:, label_index] = self._predictors[ label_index ].predict_proba( X ).reshape( n )

		S = np.sum(probability_predictions_matrix, axis=1, keepdims=True)
		return probability_predictions_matrix / S

		

	def predict( self, X ):
		"""
			Returns the predicted classes for input observations X.
		"""
		
		probability_predictions_matrix = self.predict_proba(X)

		label_indicies = np.argmax( probability_predictions_matrix, axis = 1 )

		return self.labels[ label_indicies ]










