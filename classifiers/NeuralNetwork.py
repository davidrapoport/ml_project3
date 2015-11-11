import numpy as np 

from util import sigmoid, loglikelihood, derivative_loglikelihood, gradient_descent
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
class NeuralNetwork(BaseEstimator):

	class NNLayer:
		"""
		A Neural Network Layer
		"""
		def __init__(self, layer_type, n_nodes, n_in_nodes, ith_layer):
			"""
			Parameters
			----------
			n_nodes: number of nodes 
			layer_type: string (input, hidden or output)
			n_in_nodes: number of input nodes going into each node
			ith_layer: 0: input, i-th hidden_layers, n+1: output (debugging purposes)
			"""
			
			# Each layer except the input_layer as a weight matrix used to compute
			# the dot product between the weight vector and the previous layer's
			# outputs (xs)
			# Note that each layer except the output one has a bias node

			self.n_nodes = n_nodes
			self.layer_type = layer_type
			self.ith_layer = ith_layer
			#print layer_type
			if layer_type != "output":
				self.xs = np.zeros((n_nodes+1)) #need a bias node
			else:
				self.xs = np.zeros((n_nodes))
			if layer_type != "input":
				# Randomize weights
				rand_sigma = 1 / np.sqrt(n_in_nodes) # standard deviation  
				rand_mu = 0 # mean
				self.weights = rand_sigma * np.random.randn(n_nodes,n_in_nodes+1)
				self.deltas = np.zeros((n_nodes)) # errors
			#	print "  weights", self.weights.shape
			#	print "  deltas",self.deltas.shape
			#print "  xs", self.xs.shape
				
	
	def __init__(self, n_inputs, n_outputs, n_nodes_hidden, alpha, cross_val=False, verbose=False, tol=0.001):
		"""
		Parameters
		----------
		n_outputs : number of labels (output nodes)
		n_inputs : number of training examples (+)
		n_nodes_hidden : number of nodes per hidden layer (array)
		alpha : learning rate
		eta : L2 regularization penalty parameter
		regularization : incorporates L2 penalty 
		verbose : tells you what's going on 
		"""
		self.verbose = verbose
		self.alpha = alpha
		self.n_inputs = n_inputs # add bias term
		self.n_outputs = n_outputs
		self.n_hidden_layers = len(n_nodes_hidden)
		self.n_nodes_hidden = n_nodes_hidden
		self.tol = tol
		self.epoch_accuracy = []
		self.cross_val = cross_val

		if self.verbose:
			print "Initializing input and output layers ..."
		self.input_layer = NeuralNetwork.NNLayer("input", n_inputs, 0,0) 
		self.output_layer = NeuralNetwork.NNLayer("output", n_outputs, n_nodes_hidden[-1],self.n_hidden_layers+1)
		self.hidden_layers = []
		
		if self.verbose:
			print "Initializing hidden layers ..."
		for i in range(self.n_hidden_layers):
			if i == 0: n_in_nodes = n_inputs
			else: n_in_nodes = n_nodes_hidden[i-1]
			self.hidden_layers.append(NeuralNetwork.NNLayer("hidden",n_nodes_hidden[i],n_in_nodes,i+1))


	def forwardPass(self, x):
		"""
		Given x, computes the outputs of the Network
		Parameters
		----------
		x : example 
		Returns
		-------
		nothing
		"""
		layer = self.input_layer
		layer.xs[0] = 1 # bias term
		layer.xs[1:] = x
		
		for i in range(self.n_hidden_layers):
			next_layer = self.hidden_layers[i]
			next_layer.xs[0] = 1 # bias term
			next_layer.xs[1:] = sigmoid( np.dot(next_layer.weights,layer.xs) ) 
			layer = next_layer

		self.output_layer.xs = sigmoid( np.dot(self.output_layer.weights,layer.xs) )


	def backPropagation(self, y_class):
		"""
		Propagates backwards the error shared among the NN units
		Parameters
		----------
		y_class : int in range(# nodes in output_layer)
		Returns
		-------
		nothing
		"""
		# one-hot encode y
		ol = self.output_layer
		#print "type", ol.layer_type, ", layer #", ol.ith_layer
		y = np.zeros(len(ol.xs))
		y[y_class] = 1
		# Compute correction for output units
		temp_xs = ol.xs
		temp_deltas = np.multiply(temp_xs,np.subtract(np.ones(len(temp_xs)),temp_xs))
		#print "  deltas", temp_deltas.shape
		deltas = np.multiply(temp_deltas, np.subtract(y,temp_xs))
		ol.deltas = deltas
		top_layer = ol

		# Compute correction for hidden units
		for hl in reversed(self.hidden_layers):
			#print "type", hl.layer_type, ", layer #", hl.ith_layer
			temp_deltas = np.multiply( hl.xs[1:], np.subtract(np.ones(len(hl.xs[1:])),hl.xs[1:]) ) # bias unit
			#print "  deltas", temp_deltas.shape
			#print "  weights.T, deltas",top_layer.weights.transpose().shape , top_layer.deltas.shape
			w_d = np.dot(top_layer.weights.transpose(),top_layer.deltas)[1:]
			deltas = np.multiply( temp_deltas, w_d )
			hl.deltas = deltas
			top_layer = hl

	def update_weights(self):
		"""
		Update neuron's weights according to previously computed error 
		"""
		layers = [self.output_layer] + self.hidden_layers

		for layer in layers:
			#print "type", layer.layer_type, ", layer #", layer.ith_layer
			#print "  weights", layer.weights.shape
			#print "  deltas", layer.deltas.shape
			#print "  xs", layer.xs.shape
			if layer.layer_type != "output":
				xs = layer.xs[1:]
			else: #no bias node
				xs = layer.xs
			temp = self.alpha*np.dot(layer.deltas, xs.transpose())
			layer.weights = np.add(layer.weights, temp)

	def fit(self, X, Y, X_test=None,y_test=None):
		"""Fits NeuralNetwork according to the given training data.
		Parameters
		----------
		X : shape = [n_samples, n_features]
			Training vector, where n_samples in the number of samples and
			n_features is the number of features.
		y : shape = [n_samples]
			Labels relative to X 
		Returns
		-------
		self : this NeuralNetwork object
		"""
		for i in range(50): #temporary way of checking everything works before including convergence
			for x,y in zip(X,Y): # pick a training example
				self.forwardPass(x)
				self.backPropagation(y)
				#print
				self.update_weights()
			if self.cross_val:
				y_pred = self.predict(X_test)
				accuracy = accuracy_score(y_test, y_pred)
				self.epoch_accuracy.append(accuracy)
		if self.cross_val:
			print self.epoch_accuracy
			np.save("epoch_accuracy.npy",self.epoch_accuracy)
		return self

	def predict(self, X):
		"""Predicts labels of testing data.
		Parameters
		----------
		X : shape = [n_samples, n_features]
			Testing vector, where n_samples in the number of samples and
			n_features is the number of features.
		Returns
		-------
		y_pred : shape = [n_samples]
				 Predicted Labels
		"""
		if self.verbose:
			print "Predicting labels for test data..."
		n_samples = len(X)
		n_features = len(X[0])
		y_pred = []

		for x in X:
			self.forwardPass(x)
			label = np.argmax(self.output_layer.xs)
			#print self.output_layer.xs
			y_pred.append(label)

		return y_pred

