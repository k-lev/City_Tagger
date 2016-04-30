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
from lasagne.updates import adam, sgd
import theano
import multiprocessing
from process_images import get_image_lists, prep_image_parallel

POOL_SIZE = 8


def create_vgg_net():
    '''
    Get pretrained weights from pkl file.
    Create net and use weights and biases as parameters for the layers.
    '''

    with open("/data/vgg_nolearn_saved_wts_biases.pkl") as f:
        vgg_layer_data_dict = pickle.load(f)

    vgg_nn = NeuralNet(
        layers = [
            (InputLayer, {
                        'name':'input',
                        'shape':(None,3,224,224)
                         }),
            (ConvLayer, {
                        'name':'conv1',
                        'num_filters':96,
                        'filter_size':(7,7),
                        'stride':2,
                        'flip_filters':False,
                        'W':theano.shared(vgg_layer_data_dict['conv1'][0]),
                        'b':theano.shared(vgg_layer_data_dict['conv1'][1])
                        }),
            (NormLayer, {
                        'name':'norm1',
                        'alpha':.0001
                        }),
            (PoolLayer, {
                        'name':'pool1',
                        'pool_size':(3,3),
                        'stride':3,
                        'ignore_border':False
                        }),
            (ConvLayer, {
                        'name':'conv2',
                        'num_filters':256,
                        'filter_size':(5,5),
                        'flip_filters':False,
                        'W':theano.shared(vgg_layer_data_dict['conv2'][0]),
                        'b':theano.shared(vgg_layer_data_dict['conv2'][1])
    #                     'pad':2,
    #                     'stride':1
                       }),
            (PoolLayer, {
                        'name':'pool2',
                        'pool_size':(2,2),
                        'stride':2,
                        'ignore_border':False
                        }),
            (ConvLayer, {
                        'name':'conv3',
                        'num_filters':512,
                        'filter_size':(3,3),
                        'pad':1,
    #                     'stride':1
                        'flip_filters':False,
                        'W':theano.shared(vgg_layer_data_dict['conv3'][0]),
                        'b':theano.shared(vgg_layer_data_dict['conv3'][1])
                       }),
            (ConvLayer, {
                        'name':'conv4',
                        'num_filters':512,
                        'filter_size':(3,3),
                        'pad':1,
    #                     'stride':1
                        'flip_filters':False,
                        'W':theano.shared(vgg_layer_data_dict['conv4'][0]),
                        'b':theano.shared(vgg_layer_data_dict['conv4'][1])
                        }),
            (ConvLayer, {
                        'name':'conv5',
                        'num_filters':512,
                        'filter_size':(3,3),
                        'pad':1,
    #                     'stride':1
                        'flip_filters':False,
                        'W':theano.shared(vgg_layer_data_dict['conv5'][0]),
                        'b':theano.shared(vgg_layer_data_dict['conv5'][1])
                         }),
            (PoolLayer, {
                        'name':'pool5',
                        'pool_size':(3,3),
                        'stride':3,
                        'ignore_border':False
                        }),
            (DenseLayer,{
                        'name':'fc6',
                        'num_units':4096,
                        'W':theano.shared(vgg_layer_data_dict['fc6'][0]),
                        'b':theano.shared(vgg_layer_data_dict['fc6'][1])
                       }),
            (DropoutLayer, {
                        'name':'drop6',
                        'p':.5
                        }),
            (DenseLayer, {
                        'name':'fc7',
                        'num_units':4096,
                        'W':theano.shared(vgg_layer_data_dict['fc7'][0]),
                        'b':theano.shared(vgg_layer_data_dict['fc7'][1])
                        }),
            (DropoutLayer, {
                        'name':'drop7',
                        'p':.5
                        }),
            (DenseLayer, {
                        'name':'output',
                        'num_units':3,
                        'nonlinearity':softmax,
                        })
        ],

    # #        # optimization method:
    #     update=nesterov_momentum,
    #     update_learning_rate=0.01,
        # update_momentum=0.9,

    #     #potentially ingore this
        update               = sgd,
        update_learning_rate = .05,

    #         regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=1000,  # we want to train this many epochs
        verbose=1,
        train_split=TrainSplit(eval_size=0.25),

            )

    return vgg_nn

def parallel_vectorize_images(path_prefix, image_urls, start_index, step_size, mean_image,
                                        label, label_val, pool_size, X, y, img_count):
    '''
    Input:  database collection, list of image urls (location on aws s3),
            starting index (only 1000 per call), lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Output:  array of vectorized images,
             array of vectorized labels
    '''
    # if start_index == 0 and label=='San_Francisco':
    #     return
    # if start_index == 0 and label=='London':
    #     return
    print "Parallelizing %s: from %d to %d..." %(label, start_index, start_index+step_size)
    if len(image_urls) < start_index+step_size:
        end_index = len(image_urls)
    else:
        end_index = step_size + start_index

    pool = multiprocessing.Pool(pool_size)

    img_list = pool.map(prep_image_parallel, [(path_prefix+image_url, mean_image,) for image_url in image_urls[start_index:end_index]])
    img_list = [img for img in img_list if img.size > 0]

    # vectorize image list
    X[img_count:img_count+len(img_list)] = np.concatenate(img_list,axis=0)
    y[img_count:img_count+len(img_list)] = np.array([label_val]*len(img_list))
    img_count = img_count+len(img_list)
    print label + ': finished ',start_index+step_size
    print "Total images processed:  ", img_count
    return img_count


if __name__ == '__main__':


# ****************************************************************************************
    # Begin code for pretraining data
    # Get list of images from s3
    chicago_list, london_list, sanfrancisco_list = get_image_lists()
    path_prefix = 'https://s3.amazonaws.com/rawcityimages/'

    MEAN_IMAGE = None

    #send lists, get images of correct size
    X = np.memmap('/data/X.npy', dtype='float32', mode='r+', shape=(29213,3,224,224))
    y = np.memmap('/data/y.npy', dtype='int32', mode='r+', shape=(29213))

    step_size = 1000
    img_count = 0
    for start_index in xrange(0,10000,1000):
        # if start_index > 7000:
        img_count= parallel_vectorize_images(path_prefix, london_list, start_index, step_size, MEAN_IMAGE,
                                            'London', 2, POOL_SIZE, X, y, img_count)
        img_count =parallel_vectorize_images(path_prefix, chicago_list, start_index, step_size, MEAN_IMAGE,
                                            'Chicago', 1, POOL_SIZE, X, y, img_count)
        img_count =parallel_vectorize_images(path_prefix, sanfrancisco_list, start_index, step_size, MEAN_IMAGE,
                                            'San_Francisco', 0, POOL_SIZE, X, y, img_count)

    # mean_image = X[:29213].mean(axis = 0)
    # with open('/data/training_set_mean.pkl','w') as f:
    #     pickle.dump(mean_image, f)
    #
    # X = X - mean_image
    mean_image1 = pickle.load(open('/data/training_set_mean.pkl'))

    # Shuffle the X and y arrays
    X1 = np.memmap('/data/X1.npy', dtype='float32', mode='w+', shape=(29213,3,224,224))
    y1 = np.memmap('/data/y1.npy', dtype='int32', mode='w+', shape=(29213))

    mask = np.arange(29213)
    np.random.shuffle(mask)
    for i in xrange(29213):
        X1[i] = X[mask[i]] - mean_image1
        y1 = y[mask[i]]

#************************* end pretrain data code ********************************************************
    # vgg_net = create_vgg_net()
    #
    # vgg_net.fit(X,y)
    #
    # # print vgg_net.score()
    #
    # with open('/data/trained_vgg1.pkl','w') as f:
    #     pickle.dump(vgg_net)
