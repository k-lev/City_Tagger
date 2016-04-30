import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from lasagne.updates import adam, sgd, nesterov_momentum
import theano
from process_images import get_image_lists, prep_image_parallel
from train_models import get_data

def create_nn():
    '''
    Create a neural net with one (or more) layers to fit the featurized data.
    A single softmax layer is equivalent to doing logistic regression on the featurized data.
    Result:  53% accuracy.
    Adding a fully connected hiddent layer boots accuracy to 67%.
    '''
    nn = NeuralNet(
        layers = [
            (InputLayer, {
                        'name':'input',
                        'shape':(None,4096)
                         }),
            # (DropoutLayer, {
            #             'name':'drop6',
            #             'p':.5
            #             }),
            (DenseLayer, {
                        'name':'fc7',
                        'num_units':4096,
                        }),
            (DenseLayer, {
                        'name':'output',
                        'num_units':3,
                        'nonlinearity':softmax,
                        })
                        ],
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
    #         regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=1000,  # we want to train this many epochs
        verbose=1,
        train_split=TrainSplit(eval_size=0.25),

        )

    nn.initialize()

    return nn

if __name__ == '__main__':
    X, y = get_data()
    y = y.astype(np.int32)
    print type(X[0][0])

    nn = create_nn()

    # Results:
    # Basic softmax:  53% acccuracy
    # Adding a fully connected hidden layer:  67% accuracy
    nn.fit(X, y)
