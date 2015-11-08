
# coding: utf-8

# In[1]:

from sklearn import grid_search
import numpy as np
from classifiers import LasagneNeuralNetwork
from classifiers.util import quick_save


X = np.load('./data/train_inputs.npy')
Y = np.load('./data/train_outputs.npy')

X_train = X[:int(X.shape[0]*0.95), :]
Y_train = Y[:int(X.shape[0]*0.95)]
X_test = X[int(X.shape[0]*0.95):, :]
Y_test = Y[int(X.shape[0]*0.95):]

# In[2]:

from classifiers.ImageTools import prepare_MNIST, PIXELS


# In[3]:

# data = prepare_MNIST()


# In[4]:

BasicNN = LasagneNeuralNetwork.BasicNN(input_shape=(X_train.shape, PIXELS, PIXELS))


# In[5]:

gs = grid_search.GridSearchCV(BasicNN, {'hidden_num_units':[1, 10,100], 'max_epochs':[1, 10, 20, 100]}, verbose=10000, cv=2)


# In[6]:

gs.fit(X_train, Y_train)


import cPickle as pickle

# In[25]:

pickle.dump(gs, open('./NN_gridsearch.pickle', 'w'))

# In[27]:

q = pickle.load(open('./NN_gridsearch', 'r'))

# In[28]:

print q.grid_scores_

X_kaggle = np.load('./data/test_inputs.npy')

predictions = q.predict(X_kaggle)


# In[ ]:
quick_save(predictions, append_name='_basic_nn')


