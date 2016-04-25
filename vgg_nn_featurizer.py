import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
# from nolearn.lasagne import NeuralNet
import pickle
import urllib
from resize_images import prep_image, get_bucket_files, get_aws_access, get_image_lists
import os
from pymongo import MongoClient
import json
import threading
import multiprocessing

POOL_SIZE = 8

def write_pretrained_vgg_nn_data():
    '''
    Create a vgg neural net. Load pretrained weights.
    Return the dictionary of nn layers, the lasagne output_layer, the mean_image
    '''
    # define the vgg_s network
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    # these are here JUST to upload the model; then not used
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

    # upload and input the pretrained weights
    model = pickle.load(open('./vgg_cnn_s.pkl'))
    MEAN_IMAGE = model['mean image']
    lasagne.layers.set_all_param_values(output_layer, model['values'])
    output_layer = net['fc7']
    # net.pop('drop7')
    # net.pop('fc8')
    # vgg_nn = NeuralNet(net)
    # model.pop('values')
    # with open('./vgg_data_no_wts.plk','wb') as f:
    #     pickle.dump(model, f)
    # # lasagne.layers.set_all_param_values(output_layer, model['values'])
    #
    # vgg_nn.save_params_to('./vgg_weights_only.plk')
    return net, output_layer, MEAN_IMAGE

def create_pretrained_vgg_nn():
    '''
    Create a vgg neural net. Load pretrained weights.
    Return the dictionary of nn layers, the lasagne output_layer, the mean_image
    '''
    # define the vgg_s network
    vgg_nn = NeuralNet(
        layers = [
            ('input',InputLayer),
            ('conv1',ConvLayer),
            ('norm1',NormLayer),
            ('pool1',PoolLayer),
            ('conv2',ConvLayer),
            ('pool2',PoolLayer),
            ('conv3',ConvLayer),
            ('conv4',ConvLayer),
            ('conv5',ConvLayer),
            ('pool5',PoolLayer),
            ('fc6',DenseLayer),
            ('drop6',DropoutLayer),
            ('fc7',DenseLayer)],


        #input data settings
        input_shape = (None,3,224,224),

        #convpool1 settings
        conv1_num_filters    = 96,
        conv1_filter_size = (7,7),
        conv1_pad = 0,
        conv1_stride = 2,
        conv1_flip_filters=False,
        #norm1 layer settings
        norm1_alpha = .0001,

        #pool1 layer settings
        pool1_pool_size     = (3,3),
        pool1_stride        = 3,
        pool1_ignore_border = False,

        #conv2 settings
        conv2_num_filters  = 256,
        conv2_filter_size  = (5,5),
        conv2_pad          = 2,
        conv2_stride       = 1,
        conv2_flip_filters=False,

        # pool2 settings
        pool2_pool_size     = (2,2),
        pool2_stride        = 2,
        pool2_ignore_border = False,

        #conv3 settings
        conv3_num_filters   = 512,
        conv3_filter_size   = (3,3),
        conv3_pad = 1,
        conv3_flip_filters=False,

        # conv3_stride = 1,

        #conv4 settings
        conv4_num_filters    = 512,
        conv4_filter_size = (3,3),
        conv4_pad = 1,
        conv4_stride = 1,
        conv4_flip_filters=False,

        #conv5 settings
        conv5_num_filters    = 512,
        conv5_filter_size = (3,3),
        conv5_pad = 1,
        conv5_stride = 1,
        conv5_flip_filters=False,

        # pool5 settings
        pool5_pool_size     = (3,3),
        pool5_stride        = 3,
        pool5_ignore_border = False,

        #fc6 settings
        fc6_num_units    = 4096,
        #dropout6 settings
        drop6_p = 0.5,
        #fc6 settings
        fc7_num_units    = 4096,

        #potentially ingore this
        update               = sgd,
        update_learning_rate = theano.shared(0.05),

            )

    # upload and input the pretrained weights
    model = pickle.load(open('./vgg_cnn_s.pkl'))
    MEAN_IMAGE = model['mean image']

    vgg_nn.initialize()

    return vgg_net, MEAN_IMAGE


def featurize_images(coll, path_prefix, image_urls, output_layer, mean_image, label, label_val):
    '''
    Input: database collection, list of image urls (location on aws s3),
        lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Run image throught through the network.
    Put features in the database collection
    '''
    coll.remove({})  # Remove previous entries from collection in Mongodb.

    for i, url in enumerate(image_urls):
        try:
            if i%100 == 0:
                print i, url
            # process the image
            rawim, im = prep_image(path_prefix+url, mean_image)

            # run image through NN to get features
            features = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())

            #store featues in db
            df = pd.DataFrame(features)
            json_features = df.to_json()
            data_dict = {'features':json_features, 'label':label,'label_val':label_val,'image_url':path_prefix+url}
            coll.insert_one(data_dict)

        except IOError:
            print('bad url: '+ path_prefix + url)

def featurize_one_image(coll, path_prefix, url, output_layer, mean_image, label, label_val):
    '''
    Input: database collection, list of image urls (location on aws s3),
        lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Run image throught through the network.
    Put features in the database collection
    '''

    try:
        # process the image
        rawim, im = prep_image(path_prefix+url, mean_image)

        # run image through NN to get features
        features = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())

        #store featues in db
        df = pd.DataFrame(features)
        json_features = df.to_json()
        data_dict = {'features':json_features, 'label':label,'label_val':label_val,'image_url':path_prefix+url}
        coll.insert_one(data_dict)

    except IOError:
        print('bad url: '+ path_prefix + url)


def featurize_parallel(coll_lst, path_prefix, image_lists, output_layer, mean_image, labels, label_vals, pool_size):
    print "Featurizing in parallel..."

    pool = multiprocessing.Pool(pool_size)

    pool.map(featurize_images, [(coll_lst[i], path_prefix, image_lists[i],
                                output_layer, mean_image, labels[i], label_val[i],) for i in xrange(3)])
    pool.close()
    pool.join()


if __name__ == '__main__':
    # download the pretrained model for the neural net, if not already downloaded
    if os.path.exists('./vgg_cnn_s.pkl')==False:
	    urllib.urlretrieve("https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl",
                        filename = 'vgg_cnn_s.pkl')

    # Format the original weights
    # write_pretrained_vgg_nn_data()
    # Create the neural net
    net, output_layer, MEAN_IMAGE = write_pretrained_vgg_nn_data()
    # vgg_net, MEAN_IMAGE  = create_pretrained_vgg_nn()

    # Featurize and save the data in a database
    # create MongoDB database and collection
    #******************************* THIS SCRIPT TO MAKE FEATURES AN PUT IN DB ************************
    DB_NAME = 'TRAINING_FEATURES'
    client = MongoClient()
    db = client[DB_NAME]
    coll_chi = db['Chicago']    # connect to mongodb to store scraped data
    coll_lon = db['London']    # connect to mongodb to store scraped data
    coll_sf = db['San_Francisco']    # connect to mongodb to store scraped data

    coll_lst = [coll_chi, coll_lon, coll_sf]

    path_prefix = 'https://s3.amazonaws.com/rawcityimages/'
    # image_lists = get_image_lists()
    chicago_list, london_list, sanfrancisco_list = get_image_lists()

    featurize_images(coll_chi,path_prefix, chicago_list, output_layer, MEAN_IMAGE, 'Chicago', 1)
    featurize_images(coll_lon,path_prefix, london_list, output_layer, MEAN_IMAGE, 'London', 2)
    featurize_images(coll_sf, path_prefix, sanfrancisco_list, output_layer, MEAN_IMAGE, 'San Francisco', 3)
    #******************************* END THIS SCRIPT TO MAKE FEATURES AN PUT IN DB ************************
