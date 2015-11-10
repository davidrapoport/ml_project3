
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

import theano
import lasagne


# In[3]:

from lasagne import layers
from lasagne.updates import nesterov_momentum


# In[4]:

from nolearn.lasagne import visualize
from nolearn.lasagne import NeuralNet


# In[5]:

from sklearn.metrics import classification_report, confusion_matrix

import sys

max_epochs = int(sys.argv[1])
validate_split = float(sys.argv[2])
max_split = float(sys.argv[3])

# In[6]:

X = np.load('./data/train_inputs.npy')
Y = np.load('./data/train_outputs.npy')
# Y = Y.reshape( Y.shape[0] , 1 )
Y = Y.astype(np.int32)

PIXELS = 48



print 'Original Dataset size:'
print X.shape


import generate_extra_data as ged


x_new, y_new = ged.perturb_modified_digits(X,Y,500000)
X = np.vstack((X,x_new))
Y = np.hstack((Y,y_new))

print 'New dataset size:'
print X.shape, Y.shape

X = X.reshape((-1,1, PIXELS, PIXELS))


validation_division = int(len(X)*validate_split)
top = int(len(X)*max_split)

X_train, X_val = X[:validation_division,:], X[validation_division:top,:]
Y_train, Y_val = Y[:validation_division], Y[validation_division:top]


# In[7]:

print 'Training Shape:'
print X_train.shape, Y_train.shape
print 'Validation Shape:'
print X_val.shape, Y_val.shape
print 'PIXELS:'
print PIXELS


# In[8]:

# plt.subplot(2,2,1)
# plt.imshow(X_train[0][0], cmap=cm.binary)
# # plt.imshow(X_train[0].reshape(PIXELS,PIXELS), cmap=cm.binary)
# plt.subplot(2,2,2)
# plt.imshow(X_train[2][0], cmap=cm.binary)
# # plt.imshow(X_train[2].reshape(PIXELS, PIXELS), cmap=cm.binary)
# plt.subplot(2,2,3)
# plt.imshow(X_train[4][0], cmap=cm.binary)
# # plt.imshow(X_train[4].reshape(PIXELS, PIXELS), cmap=cm.binary)
# plt.subplot(2,2,4)
# plt.imshow(X_train[8][0], cmap=cm.binary)
# # plt.imshow(X_train[8].reshape(PIXELS, PIXELS), cmap=cm.binary)


# In[9]:

# Implementation of L2 Regularization
def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses



net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('maxpool1', layers.MaxPool2DLayer),
        # ('dropout1', layers.DropoutLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('maxpool2', layers.MaxPool2DLayer),
        ('conv2d3', layers.Conv2DLayer),
        ('maxpool3', layers.MaxPool2DLayer),

        ('conv2d4', layers.Conv2DLayer),
        ('maxpool4', layers.MaxPool2DLayer),
        
        ('dense1', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer)
    ],
    # input layer descriptors
    input_shape=(None, 1, PIXELS, PIXELS),


    # convolution layer descriptors
    conv2d1_num_filters=16,
    conv2d1_filter_size=(5,5),
    conv2d1_nonlinearity = lasagne.nonlinearities.rectify,
    conv2d1_W = lasagne.init.GlorotUniform(),
    
    # maxppol layer descriptors
    maxpool1_pool_size=(2,2),

    # dropout

    # dropout1_p = 0.5,

    # convolution layer descriptors
    conv2d2_num_filters=32,
    conv2d2_filter_size=(3,3),
    conv2d2_nonlinearity = lasagne.nonlinearities.rectify,
    conv2d2_W = lasagne.init.GlorotUniform(),
    
    # maxppol layer descriptors
    maxpool2_pool_size=(2,2),
    
    # convolution layer descriptors
    conv2d3_num_filters=64,
    conv2d3_filter_size=(3,3),
    conv2d3_nonlinearity = lasagne.nonlinearities.rectify,
    conv2d3_W = lasagne.init.GlorotUniform(),

    maxpool3_pool_size=(2,2),




    # convolution layer descriptors
    conv2d4_num_filters=128,
    conv2d4_filter_size=(3,3),
    conv2d4_nonlinearity = lasagne.nonlinearities.rectify,
    conv2d4_W = lasagne.init.GlorotUniform(),
    
    # maxppol layer descriptors
    maxpool4_pool_size=(2,2),

    dense1_num_units=128,

    # dropout layer descriptors
    dropout2_p = 0.5,
    
    # output layer descriptors
    output_nonlinearity = lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    #optimization parameters
    update=nesterov_momentum,
    update_learning_rate=0.01,
    max_epochs=max_epochs,
    verbose=1000000
)


# print net.layers_
# In[10]:
print('training')
nn = net.fit(X_train, Y_train)
print('training complete')

# In[11]:



# In[12]:

X_test = np.load('./data/test_inputs.npy')


# In[13]:

X_test = X_test.reshape(-1, 1, PIXELS, PIXELS)


# In[14]:
print('predicting...')
predicted = nn.predict(X_test)


# In[15]:

indicies = [x for x in range(1,len(predicted)+1)]
submission = pd.DataFrame(indicies, columns=['Id']).join(pd.DataFrame(predicted, columns=['Prediction']))

submission.to_csv('predicted_4cnn_epochs'+str(max_epochs)+'_valsplit'+str(validate_split)+'.csv',index=False)


# In[16]:
print('predicting done.')
Y_pred = nn.predict(X_val)


# In[17]:

# plt.imshow(confusion_matrix(Y_val, Y_pred), interpolation='nearest')


# In[18]:
print 'PARAMS:', nn.get_params()
print 'nn.score:\n', nn.score(X_val, Y_val)
print 'confusion_matrix:\n',confusion_matrix(Y_val, Y_pred)

import cPickle as pickle

pickle.dump(nn, open('./QuadCNN.pickle', 'w'))
print('Pickle saved')
print('terminate')

# In[19]:

# import random
# wrong_images_ids = np.where(Y_pred.astype(np.int32) != Y_val.astype(np.int32))[0]
# selected_images_ids = random.sample(wrong_images_ids, 8)
# plt.figure(figsize=(10,11))
# for id, image_id in enumerate(selected_images_ids):
#     plt.subplot(4,4,id+1)
#     plt.imshow(X_val[image_id][0], cmap=cm.binary)
#     plt.title(''.join(['is: ', str(Y_val[image_id]), ' predicted:', str(Y_pred[image_id])]))


# In[20]:

# visualize.plot_conv_weights(nn.layers_['conv2d1'])


# In[ ]:



