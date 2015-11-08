
# coding: utf-8

# In[1]:

from sklearn import grid_search
import numpy as np
from classifiers import LasagneNeuralNetwork
from classifiers.util import quick_save
from classifiers.ImageTools import PIXELS

X_train = np.load('./data/train_inputs.npy')
Y_train = np.load('./data/train_outputs.npy')



# In[2]:

X_train = X_train.reshape(X_train.shape[0], PIXELS, PIXELS)
Y_train = Y_train.reshape()
# In[3]:

# data = prepare_MNIST()


# In[4]:

BasicNN = LasagneNeuralNetwork.BasicNN(input_shape=(None, PIXELS, PIXELS))


# In[5]:

gs = grid_search.GridSearchCV(BasicNN, {'hidden_num_units':[100], 'max_epochs':[1, 10, 20, 100], 'update_learning_rate':[0.1,0.01,0.001]}, verbose=10000, cv=2, num_jobs=8)


# In[6]:

gs.fit(X_train, Y_train)


import cPickle as pickle

# In[25]:

pickle.dump(gs, open('./NN_gridsearch.pickle', 'w'))

# In[27]:

q = pickle.load(open('./NN_gridsearch.pickle', 'r'))

# In[28]:

print q.grid_scores_

X_kaggle = np.load('./data/test_inputs.npy')

X_kaggle = X_kaggle.reshape(X_kaggle.shape[0], PIXELS, PIXELS)

predictions = q.predict(X_kaggle)


# In[ ]:
quick_save(predictions, append_name='_basic_nn')


