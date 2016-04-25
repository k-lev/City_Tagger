import numpy as np
import matplotlib.pyplot as plt
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
import cPickle as pickle
import urllib
from resize_images import prep_image

# Code heavily borrowed from:
# https://github.com/Lasagne/Recipes/blob/master/examples/ImageNet%20Pretrained%20Network%20%28VGG_S%29.ipynb
# lasange recipies in the lasagne/recipies github
# VGG_cnn pretrained on Imagenet
#
#This example demonstrates using a network pretrained on ImageNet for classification. The model used was
#   converted from the VGG_CNN_S model (http://arxiv.org/abs/1405.3531) in Caffe's Model Zoo.
# License:
#   The model is licensed for non-commercial use only

def create_pretrained_vgg_nn_test():

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
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']

    # upload and input the pretrained weights
    model = pickle.load(open('./vgg_cnn_s.pkl'))
    CLASSES = model['synset words']
    MEAN_IMAGE = model['mean image']
    lasagne.layers.set_all_param_values(output_layer, model['values'])

    return net, output_layer, CLASSES, MEAN_IMAGE

def test_net_images(num_images=5):
    '''
    Input:  number of images to test net on.
    Output: list of image urls, num_images == length
    '''

    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')

    np.random.seed(23)
    np.random.shuffle(image_urls)
    image_urls = image_urls[:num_images]

    return image_urls

def process_test_images(image_urls, output_layer, mean_image, CLASSES):
    '''
    Input: list of image urls
    Process each image (vectorize, resize and crop).
    Run image throught through the network.
    Get predictions.  Show top five results next to image.
    '''

    for url in image_urls:
        try:
            rawim, im = prep_image(url, mean_image)

            prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
            top5 = np.argsort(prob[0])[-1:-6:-1]

            plt.figure()
            plt.imshow(rawim.astype('uint8'))
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, CLASSES[label]), fontsize=14)
            plt.show()
        except IOError:
            print('bad url: ' + url)
