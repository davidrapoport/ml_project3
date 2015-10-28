"""
	Utillity functions common to classifiers
"""

import numpy as np

def sigmoid( a ):
	"""	
		Computes the logistic/signmoid function element wise
		@params
			a: exponent 
		@return: 1 / 1 + exp( -a )
	"""

	return 1 / ( 1 + np.exp( -a ) )

def loglikelihood( X, Y, w ):
	"""	
		Returns the loglikelihood function 
		@params:
			X: data matrix of size obersvations * features 
			(ie.  ( n, m ) )
			Y: labels of shape ( n, 1 )
			w: weight vector of shape ( m,  1 )
		@returns:
			loglikelihood function.
	"""
	# note sure if this is necessary right now...

	return -Y.T * np.log( sigmoid( X * w ) ) - ( 1 - Y ).T * np.log( 1 - sigmoid( X * w ) )
	

def derivative_loglikelihood( X, Y, w ):
	"""
		Returns the derivative of the loglikelihood function
		@params:
			X: data matrix of size obersvations * features 
			(ie.  ( n, m ) )
			Y: labels of shape ( n, 1 )
			w: weight vector of shape ( m,  1 )
		@returns:
			dL(w)/dw the derivative of the loglikelihood function			
	"""

	return - np.dot( X.T,  Y - sigmoid( np.dot( X, w ) )  )


def gradient_descent( X, Y, start_weights = False, \
	error_derivative = derivative_loglikelihood, \
	max_iterations = 100000, \
	alpha = 0.001, \
	return_weights = False ):
	"""
		Computes the weights of a linear function using gradient descent
		@params:
			X: data matrix of size obersvations * features 
			(ie.  ( n, m ) )
			Y: labels of shape ( n, 1 )
			start_weights[False]: A starter weight vector ( m, 1 ). If not supplied a random weight is picked.
			error_derivative: the derivative of an error function, be default we use the derivative of loglikelihood
			max_iterations[10000]: the maximum number of iterations to conduct gradient descent for 
			alpha[0.001]: learning rate, or increment with which we change the weights
			return_weights[False]: returns the weights computed after each iteration as an array.			
	"""
	# number of observations , number of weights
	n, m = X.shape
	assert Y.shape[0] == n, 'Y must have the same number of observations as X'

	# check if starter weights have been provided
	if type( start_weights ) != bool:
		assert start_weights.shape[0] == m, 'start_weights must have the same length as features in X'
		w = start_weights
	else:
		# not provided, initialize random weights
		w = np.random.random( size = m ).reshape( m, 1 )
	#endif

	if return_weights:
		w_iterated = [ w ]
	#endif

	for iteration in xrange(max_iterations):

		# update step
		w_new = w - alpha * error_derivative( X, Y, w )

		if return_weights:
			w_iterated.append(w_new)
		#endif

		# @TODO: do I really need to create, w and w_new?
		w = w_new
	#endfor

	if return_weights:
		return w, w_iterated
	else:
		return w
	#endif







