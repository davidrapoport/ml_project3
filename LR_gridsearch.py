import numpy as np
from sklearn import grid_search
# get_ipython().magic(u'pylab inline')
from classifiers.Logistic import LogisticRegression

X = np.load('./data/train_inputs.npy')
Y = np.load('./data/train_outputs.npy')

X_train = X[:int(X.shape[0]*0.95), :]
Y_train = Y[:int(X.shape[0]*0.95)]
X_test = X[int(X.shape[0]*0.95):, :]
Y_test = Y[int(X.shape[0]*0.95):]

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


lr = LogisticRegression(learning_rate=0.0005,max_iterations=100000)


# In[11]:

gs = grid_search.GridSearchCV(lr, {'learning_rate':[0.01, 0.001, 0.05, 0.005 ], 'max_iterations':[10, 100, 1000, 10000, 5000, 100000]}, verbose=10000)

# gs = grid_search.GridSearchCV(lr, {'learning_rate':[0.01, 0.001, 0.1], 'max_iterations':[10, 100, 1000, 10000, 100000]}, verbose=10000)

# gs = grid_search.GridSearchCV(lr, {'learning_rate':[0.01], 'max_iterations':[10, 100]}, verbose=10000)



# In[12]:

gs.fit(X_train,Y_train)


# In[24]:

import cPickle as pickle


# In[25]:

pickle.dump(gs, open('./LR_gridsearch.pickle', 'w'))


# In[27]:

q = pickle.load(open('./LR_gridsearch', 'r'))


# In[28]:

print q.grid_scores_


# In[ ]:



