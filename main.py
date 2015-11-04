import os, pdb
import numpy as np 
from convert_to_numpy import convert_all

if not os.path.exists("data/train_inputs.npy"):
	convert_all()

train_inputs = np.load("data/train_inputs.npy")
train_targets = np.load("data/train_outputs.npy")
print str(train_inputs.shape)
print str(train_targets.shape)