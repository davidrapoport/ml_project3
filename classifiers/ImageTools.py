import random
import numpy as np
import matplotlib.pyplot as plt

PIXELS = 48

def show_wrong_images(X, Y_true, Y_pred):
	"""
		this function plots some wrong images that were misclassified
		@params:
			X: the list of images of shape, (n, PIXEL, PIXEL)
			Y_true: the true labels for the images
			Y_pred: the predicted labels for the images
	"""
	wrong_images_ids = np.where(Y_pred.astype(np.int32) != Y_true.astype(np.int32))[0]
	selected_images_ids = random.sample(wrong_images_ids, 8)
	plt.figure(figsize=(10,11))
	for id, image_id in enumerate(selected_images_ids):
		plt.subplot(4,4,id+1)
		try:
			plt.imshow(X[image_id], cmap=cm.binary)
		except Exception as e:
			try:
				plt.imshow(X[image_id][0], cmap=cm.binary)
			except Exception as e:
				raise Exception('Failed to display image')
			#endtry
		#endtry
		plt.title(''.join(['is: ', str(Y_true[image_id]), ' predicted:', str(Y_pred[image_id])]))



def prepare_MNIST(x_location='./data/train_inputs.npy', y_location='./data/train_outputs.npy', PIXELS=48, validation_split=0.8, max_split=1):

	assert validation_split < max_split, 'the max_split has to be greater than the validation_split'

	X = np.load(x_location)
	Y = np.load(y_location)
	Y = Y.astype(np.int32)

	validation_division = int(len(X)*validation_split)
	top = int(len(X)*max_split)

	X = X.reshape(X.shape[0], 1, PIXELS, PIXELS)

	X_train, X_val = X[:validation_division,:], X[validation_division:top,:]
	Y_train, Y_val = Y[:validation_division], Y[validation_division:top]

	return { 'X_train' : X_train , 'Y_train' : Y_train , 'X_val' : X_val , 'Y_val' : Y_val }
