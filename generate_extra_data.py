import numpy as np 
from scipy.signal import convolve2d
from scipy.misc import imresize, imfilter, imread
from scipy.ndimage.interpolation import rotate
# from skimage.transform import warp, AffineTransform
from matplotlib import pyplot as plt
import pdb

patterns = []
for i in range(6):
	pat = imread("data/patternpics/texture%s.jpg"%i, flatten=True)
	pat = imresize(pat,(48,48))
	# plt.imshow(pat, cmap="Greys_r")
	# plt.show()
	patterns.append(pat)

def generate_extra_data(n=3000, file_name=None, load_file=None, seed=None):
	if load_file:
		if not load_file.startswith("data/"):
			load_file = "data/" + load_file
		return np.load(load_file +".npy"), np.load(load_file+"_targets.npy")
	if seed:
		np.random.seed(seed)
	X = np.load("data/mnist_train_extra_inputs.npy")
	Y = np.load("data/mnist_train_extra_outputs.npy")
	positions = np.random.random_integers(0, X.shape[0], n)
	transformed = X[positions,:]
	targets = Y[positions]

	final = np.zeros((n,48*48))
	angles = np.random.uniform(0,359, n)
	for i in range(n):
		final[i,:] = imfilter(imresize(rotate(transformed[i,:].reshape((28,28)), angles[i]), (48,48)), "emboss").flatten()
	if file_name:
		if not file_name.startswith("data/"):
			file_name = "data/" + file_name
		np.save(file_name, final)
		np.save(file_name + "_targets", targets)
	return final, targets


def show_image(ex, size=(48,48)):
	im = np.reshape(ex, size)
	plt.imshow(im, cmap="Greys_r")
	plt.show()

# X = np.load("data/mnist_train_extra_inputs.npy")
# show_image(X[4,:], (28,28))
# ex = X[4,:].reshape((28,28))
# ex = warp(ex.astype(float), AffineTransform(shear=3.14))
# show_image(ex.flatten(), (28,28))

