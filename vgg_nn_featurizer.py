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
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import urllib
from process_images import prep_image, get_bucket_files, get_aws_access, get_image_lists, write_to_s3, prep_image_parallel
import os
from pymongo import MongoClient
import json
import multiprocessing

POOL_SIZE = 8

def create_pretrained_vgg_nn_data():
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
    net.pop('fc8')

    return net, output_layer, MEAN_IMAGE

def create_pretrained_vgg_nn_nolearn():
    '''
    Create a vgg neural net. Load pretrained weights.
    Return the dictionary of nn layers, the lasagne output_layer, the mean_image
    '''
    # define the vgg_s network
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
                        'flip_filters':False
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
                        'flip_filters':False
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
                        'flip_filters':False
                       }),
            (ConvLayer, {
                        'name':'conv4',
                        'num_filters':512,
                        'filter_size':(3,3),
                        'pad':1,
    #                     'stride':1
                        'flip_filters':False
                        }),
            (ConvLayer, {
                        'name':'conv5',
                        'num_filters':512,
                        'filter_size':(3,3),
                        'pad':1,
    #                     'stride':1
                        'flip_filters':False
                         }),
            (PoolLayer, {
                        'name':'pool5',
                        'pool_size':(3,3),
                        'stride':3,
                        'ignore_border':False
                        }),
            (DenseLayer,{
                        'name':'fc6',
                        'num_units':4096
                       }),
            (DropoutLayer, {
                        'name':'drop6',
                        'p':.05
                        }),
            (DenseLayer, {
                        'name':'fc7',
                        'num_units':4096
                        }),
        ],



    #        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

    #     #potentially ingore this
    #     update               = sgd,
    #     update_learning_rate = theano.shared(0.05),

    #         regression=True,  # flag to indicate we're dealing with regression problem
    #     max_epochs=400,  # we want to train this many epochs
    #     verbose=1,

            )

    # upload pretrained weights
    vgg_nn.initialize()
    vgg_nn.load_params_from('./vgg_nolearn_saved_wts_biases.pkl')

    # upload mean image
    model = pickle.load(open('./vgg_cnn_s.pkl'))
    mean_image = model['mean image']
    with open("/data/mean_image.pkl", 'w') as f:
        pickle.dump(mean_image, f)
    # pickel the model and the mean image
    with open("/data/full_vgg.pkl", 'w') as f:
        pickle.dump(vgg_nn, f)

    return vgg_net, mean_image

def load_precreated_vgg_nn_and_mean_img():
    '''
    Retreive fully initialized nolean vgg neural net mode from file.
    Retreive mean image data from file.
    Return:  nolearn NeuralNet, mean_image array
    '''

    with open("/data/mean_image.pkl") as f_un:
        mean_image = pickle.load(f_un)
    with open("/data/full_vgg.pkl") as f:
        vgg_nn = pickle.load(f)

        return vgg_nn, mean_image


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






def featurize_1000images_pkl(path_prefix, image_urls, start_index, step_size, output_layer, mean_image, label, label_val):
    '''
    Input: database collection, list of image urls (location on aws s3), starting index (only 1000 per call)
        lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Run image throught through the network.
    Put the features in a dataframe.
    Pickle and save the datframe locally and to s3
    '''
    # if start_index == 0 and label=='San_Francisco':
    #     return
    # if start_index == 0 and label=='London':
    #     return

    df = pd.DataFrame()
    if len(image_urls) < start_index+step_size:
        end_index = len(image_urls)
    else:
        end_index = step_size + start_index
    df_list = []
    for i, url in enumerate(image_urls[start_index:end_index]):
        try:
            # if i%100 == 0:
            print i+start_index, url
            # process the image
            rawim, im = prep_image(path_prefix+url, mean_image)
            # run image through NN to get features
            features = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
            #store featues in df
            current_img_df = pd.DataFrame(features)
            current_img_df['label'] = label
            current_img_df['label_val'] = label_val
            current_img_df['url'] = url
            df_list.append(current_img_df)
            # df = pd.concat([df, current_img_df])

        except IOError:
            print('bad url: '+ path_prefix + url)

    # after finished, pickle it
    df = pd.concat(df_list)
    filepath_filename = 'data/'+label+str(end_index)+'.pkl'
    print filepath_filename
    df.to_pickle(filepath_filename)
    write_to_s3('featuresbucket', filepath_filename)

def parallel_featurize_1000images_pkl(path_prefix, image_urls, start_index, step_size, vgg_nn, mean_image,
                                        label, label_val, pool_size):
    '''
    Input: database collection, list of image urls (location on aws s3), starting index (only 1000 per call)
        lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Run image throught through the network.
    Put the features in a dataframe.
    Pickle and save the datframe locally and to s3
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
    print start_index, end_index
    df_list = []

    pool = multiprocessing.Pool(pool_size)

    img_list = pool.map(prep_image_parallel, [(path_prefix+image_url, mean_image,) for image_url in image_urls[start_index:end_index]])

    img_list = [img for img in img_list if img.size > 0]

    # vectorize image list
    image_vectors = np.concatenate(img_list,axis=0)

    featurized_images = vgg_nn.predict_proba(image_vectors)

    df = pd.DataFrame(featurized_images)
    df['label'] = label
    df['label_val'] = label_val

    filepath_filename = '/data/'+label+str(end_index)+'.pkl'
    print filepath_filename
    df.to_pickle(filepath_filename)
    write_to_s3('featuresbucket', filepath_filename)



def featurize_one_image(coll, path_prefix, url, output_layer, mean_image, label, label_val):
    '''
    Input: database collection, list of image urls (location on aws s3),
        lasagne output_layer, mean_image, imageset label, imageset lavel_val (int)
    Process each image (vectorize, resize and crop).
    Run image through through the network.
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

def featurize_image(full_path_filename, output_layer, mean_image):
    '''
    Input: full path to the location of the file,
        lasagne output_layer, mean_image from the training data
    Process the image (vectorize, resize and crop).
    Run image through through the network.
    Output:  features array shape: (4096,1)
    '''

    try:
        # process the image
        rawim, im = prep_image(path_prefix+url, mean_image)

        # run image through NN to get features
        features = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())

        return features

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
    # create_pretrained_vgg_nn_data()
    # Create the neural net
    # net, output_layer, MEAN_IMAGE = create_pretrained_vgg_nn_data()
    # vgg_net, MEAN_IMAGE  = create_pretrained_vgg_nn()
    vgg_net, MEAN_IMAGE  = load_precreated_vgg_nn_and_mean_img()


    path_prefix = 'https://s3.amazonaws.com/rawcityimages/'

    # image_lists = get_image_lists()
    chicago_list, london_list, sanfrancisco_list = get_image_lists()

    #******************** Begin script:  pickle features ****************************

    step_size = 1000
    for start_index in xrange(0000,10000,1000):
        parallel_featurize_1000images_pkl(path_prefix, london_list, start_index, step_size, vgg_net, MEAN_IMAGE, 'London', 2, POOL_SIZE)
        parallel_featurize_1000images_pkl(path_prefix, chicago_list, start_index, step_size, vgg_net, MEAN_IMAGE, 'Chicago', 1, POOL_SIZE)
        parallel_featurize_1000images_pkl(path_prefix, sanfrancisco_list, start_index, step_size, vgg_net, MEAN_IMAGE, 'San_Francisco', 3, POOL_SIZE)
