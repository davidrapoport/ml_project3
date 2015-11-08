import numpy as np
import pandas as pd
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import visualize
from nolearn.lasagne import NeuralNet
from sklearn.metrics import classification_report, confusion_matrix

#constants
PIXELS = 48

def BasicNN(max_epochs=100, hidden_num_units=100, update_learning_rate=0.01, input_shape=(None, PIXELS, PIXELS)):
	net = NeuralNet(
	    layers=[
	        ('input', layers.InputLayer),
	        ('hidden', layers.DenseLayer),
	        ('output', layers.DenseLayer)
	    ],
	    input_shape=input_shape,

	    hidden_num_units=100,

	    output_nonlinearity = lasagne.nonlinearities.softmax,
	    output_num_units=10,

	    #optimization parameters
	    update=nesterov_momentum,
	    update_learning_rate=update_learning_rate,
	    max_epochs=max_epochs
	)

	return net


def DoubleCNN(max_epochs=100, update_learning_rate=0.01, input_shape=(None, 1, PIXELS, PIXELS)):
	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv2d1', layers.Conv2DLayer),
			('maxpool1', layers.MaxPool2DLayer),
			#         ('dropout1', layers.DropoutLayer),
			('conv2d2', layers.Conv2DLayer),
			('maxpool2', layers.MaxPool2DLayer),
			('dropout2', layers.DropoutLayer),
			('output', layers.DenseLayer)
		],
		# input layer descriptors
		input_shape=(None, 1, PIXELS, PIXELS),


		# convolution layer descriptors
		conv2d1_num_filters=32,
		conv2d1_filter_size=(5,5),
		conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
		conv2d1_W = lasagne.init.GlorotUniform(),

		# maxppol layer descriptors
		maxpool1_pool_size=(2,2),

		# dropout layer descriptors
		#     dropout1_p = 0.5,

		# convolution layer descriptors
		conv2d2_num_filters=32,
		conv2d2_filter_size=(5,5),
		conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
		#     conv2d2_W = lasagne.init.GlorotUniform(),

		# maxppol layer descriptors
		maxpool2_pool_size=(2,2),

		# dropout layer descriptors
		dropout2_p = 0.5,

		# output layer descriptors
		output_nonlinearity = lasagne.nonlinearities.softmax,
		output_num_units=10,

		#optimization parameters
		update=nesterov_momentum,
		update_learning_rate=0.01,
		max_epochs=10
	)

	return net
