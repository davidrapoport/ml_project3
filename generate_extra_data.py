import numpy as np 
from scipy.signal import convolve2d
from scipy.misc import imresize, imfilter, imread
from scipy.ndimage.interpolation import rotate as imrotate
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d
# from skimage.transform import warp, AffineTransform
from matplotlib import pyplot as plt
import pdb

# patterns = []
# for i in range(6):
# 	pat = imread("data/patternpics/texture%s.jpg"%i, flatten=True)
# 	pat = imresize(pat,(48,48))
# 	# plt.imshow(pat, cmap="Greys_r")
# 	# plt.show()
# 	patterns.append(pat)


def generate_extra_data(n=3000, file_name=None, load_file=None, seed=None, rotate=True, emboss=True, elastic=False):
	print "Creating {} new images".format(n)
	if load_file:
		if not load_file.startswith("data/"):
			load_file = "data/" + load_file
		return np.load(load_file +".npy"), np.load(load_file+"_targets.npy")
	if seed:
		np.random.seed(seed)
	X = np.load("data/mnist_train_extra_inputs.npy")
	Y = np.load("data/mnist_train_extra_outputs.npy")
	positions = np.random.random_integers(0, X.shape[0]-1, n)
	transformed = X[positions,:]
	targets = Y[positions]
	final = np.zeros((n,48*48))
	angles = np.random.uniform(0,359, n)
	for i in range(n):
		temp = transformed[i,:].reshape((28,28))
		temp = imresize(temp, (48,48))
		if rotate:
			temp = imrotate(temp, angles[i], reshape=False)
		if emboss:
			temp = imfilter(temp,"emboss")
		if elastic:
			temp = elastic_distortion(temp, size=48)
		final[i,:] = temp.flatten()
	if file_name:
		if not file_name.startswith("data/"):
			file_name = "data/" + file_name
		np.save(file_name, final)
		np.save(file_name + "_targets", targets)
	return final, targets


def show_image(ex, size=(48,48), name=None, norm=0.0):
	fig = plt.figure()
	im = np.reshape(ex, size)
	plt.imshow(im, cmap="Greys_r")
	# plt.show()
	if name:
		plt.title(name)
		if norm:
			plt.suptitle("Norm of difference="+str(norm))
		fig.savefig("".join(name.split(" ")))
	return plt

def perturb_modified_digits(X, Y, n=3000, seed=None, alpha=8, sigma=3):
	print "Modifying {} digits".format(n)
	positions = np.random.random_integers(0, X.shape[0]-1, n)
	transformed = X[positions,:]
	targets = Y[positions]

	final = np.zeros((n,48*48))
	angles = np.random.uniform(0,359, n)
	for i in range(n):
		temp = transformed[i,:].reshape((48,48))
		temp = imrotate(temp, angles[i], reshape=False)
		temp = elastic_distortion(temp, size=48, alpha=alpha, sigma=sigma)
		final[i,:] = temp.flatten()
	return final, targets


def elastic_distortion(ex, size=28, alpha=8, sigma=3):

	Xdis = np.random.uniform(-1,1,(size*size)).reshape((size,size))
	Ydis = np.random.uniform(-1,1,(size*size)).reshape((size,size))

	Xdis = gaussian_filter(Xdis, sigma=sigma, mode="constant")
	Ydis = gaussian_filter(Ydis, sigma=sigma, mode="constant")
	Xdis = (Xdis/ np.linalg.norm(Xdis)) * alpha
	Ydis = (Ydis/ np.linalg.norm(Ydis)) * alpha
	ex_new = np.zeros((size,size))
	k=0
	for i in range(size):
		for j in range(size):
			s=0
			xp, scale_x = divmod(Xdis[i, j] + i, 1)
			yp, scale_y = divmod(Ydis[i, j] +j,1)
			xp, yp = int(xp), int(yp)
			try:
				ex_yp_xp=ex[yp,xp] 
			except Exception, e:
				ex_yp_xp = 0
				s += 1
			try:
				ex_yp_xp1 = ex[yp, xp+ 1]
			except Exception, e:
				ex_yp_xp1 = 0
				s += 1
			try:
			 	ex_yp1_xp1 = ex[yp+1, xp+ 1]
			except Exception, e:
				ex_yp1_xp1 = 0
				s += 1
			try:
				ex_yp1_xp = ex[yp+1,xp]
			except Exception, e:
				s += 1
				ex_yp1_xp = 0
			# new_x1 = ex[yp, xp] + scale_x*(ex[yp, xp+ 1] - ex[yp,xp])
			# new_x2 = ex[yp+1, xp] + scale_x*(ex[yp+1, xp+ 1] - ex[yp+1,xp])
			# new_val = new_x1 +scale_y*(new_x2 - new_x1)
			new_x1 = ex_yp_xp + scale_x*(ex_yp_xp1 - ex_yp_xp)
			new_x2 = ex_yp1_xp + scale_x*(ex_yp1_xp1 - ex_yp1_xp)
			new_val = new_x1 + scale_y*(new_x2 - new_x1)
			ex_new[j,i] = new_val
			if s:
				k += 1
	# print float(k)/(size*size)

	return ex_new

if __name__ == '__main__':
	X = np.load("data/mnist_train_extra_inputs.npy")
	original = show_image(X[4,:], (28,28), name="Original MNIST")
	e,_ = generate_extra_data(n=4)
	for cnt in range(4):
		show_image(e[cnt,:], name="Extra data "+str(cnt+1))
	ex = X[4,:].reshape((28,28))

	# sig3alph8 = elastic_distortion(ex, sigma=3, alpha=8)
	# show_image(sig3alph8.flatten(), (28,28), name="Sigma=3, Alpha=8", norm=np.linalg.norm(sig3alph8 - ex))
	# show_image((sig3alph8 - ex).flatten(), (28,28))
	# print np.linalg.norm(sig3alph8 - ex)

	# sig001 = elastic_distortion(ex, sigma=0.01)
	# show_image(sig001.flatten(), (28,28), name="Sigma=1e-2, Alpha=8", norm=np.linalg.norm(sig001 - ex))
	# show_image((sig001 - ex).flatten(), (28,28))
	# print np.linalg.norm(sig001 - ex)

	# alph30 = elastic_distortion(ex, alpha=30)
	# show_image(alph30.flatten(), (28,28), name="Sigma=3, Alpha=30", norm=np.linalg.norm(alph30 - ex))
	# show_image((alph30 - ex).flatten(), (28,28))
	# print np.linalg.norm(alph30 - ex)

	# ex_new = elastic_distortion(ex, alpha = 0.01)
	# show_image(ex_new.flatten(), (28,28))

	# ex_new = elastic_distortion(ex, sigma=3)
	# show_image(ex_new.flatten(), (28,28))

