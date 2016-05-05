# City Tagger

## Contents
+ [Introduction](#intro)
+ [Motivation](#motivation)
+ [Data](#data)
+ [The Big Technical Challenge](#challenge)
+ [The Pipeline](#pipeline)
+ [Results](#results)
+ [Sample Webpage Results](#sample)
+ [Package Requirements](#requirements)
+ [Instructions](#instructions)
+ [Credits](#Credits)
+ [License](#license)

## <a name="intro">Introduction

City Tagger automatically identifies the city where your pictures were taken using only the features found in the images themselves.  As a proof of concept, the model is limited to identifying Chicago, London, and San Francisco.

The model is incorporated into a web app, where users can upload an image or input a url and the model will identify the city where the image was taken.

## <a name="motivation">Motivation

The untagged photo problem has both a retail and wholesale version.

Many of us have been in the position of uploading hundreds of vacation pictures to our computers and then tagging them by hand.  It's an annoying time-suck.  That's the retail version.

The wholesale version is much more serious.  Flickr has over 10 billion images, and only 3-4% have any geotagging information.  They are actively researching how to solve this problem, as they want it both for their users and their business model.

## <a name="Data">Data
I downloaded 30,000 images from Flickr using FlickrApi, 10,000 from each of the three cities, and stored them in an AWS s3.

Be aware if doing this yourself, that the actual process involved downloading 43,000 images and rejecting 13,000 by hand.  Your Flickr search results are only as good as the image tags, and if a user has pictures of Paris tagged as "London to Paris Trip", you'll get these as part of your search for "London."  

## <a name="challenge">The Big Technical Challenge

Image data has far too many features for standard machine learning techniques to categorize.  An image whose dimensions are 224x224 (x 3 colors) has 150,528 features.  What's more, the features are highly correlated and redundant... in other words, they are largely bad features.

The solution I can up with was to featurize my images by passing them through a deep, pretrained, convolutional neural network.  Deep, convolutional neural nets are designed to be adept at finding complex, non-linear image features in images while making excellent assumptions about what kinds of features are relevant.  For example, they care about the localized relationships between pixels, but ignore the absolute positions of those features in the image.    

What's more what deep, convolutional neural networks learn about images, if trained on a rich enough dataset, his highly transferrable to new datasets.

The essential point for the City Tagger project is that I can pass an image into the neural net with 150,000 (largely bad) features, and it will output 4096 excellent image features, a number quite tractable with standard machine learning techniques.

## <a name="pipeline">The Pipeline

1.  Scrape 30,000 clean images and store them in s3
2.  Retrieve an image from s3
3.  Preprocess the image (resize, reshape, normalize using mean neural net training image)
4.  Featurize the image by passing it through the neural networks
5. Train an SVM model using the featurized images.

(Note: steps 2-5 are highly parallelized.)

## <a name="results">Results

The model has 73.5% accuracy.

## <a name="sample">Sample Webpage Sample Result

![Result Sample](https://raw.github.com/k-lev/City_Tagger/master/img/webPageSample.png)

## <a name="requirement">Package Requirements
pandas
numpy
matplotlib
Scikit-Learn
Scipy
Flask
lasagne
theano
nolearn
FlickrApi

## <a name="instructions">Instructions

1. Set up an s3 account on AWS or have a LOT of memory available.
2. Make sure you have set up all requirements.
3. Run scrape_images.py
4. Delete images that are not relevant.  Judging by my experience, you will only keep about 70% of your images.
5.  in vgg_nn_featurizer.py
    - Run create_pretrained_vgg_nn_data() to retrieve pretrained weights and biases for the vgg-s neural network and put them in a format that a nolearn.NeuralNet can read.
    - Run create_pretrained_vgg_nn() to create and store the nolearn.NeuralNet instance.
6. Run vgg_nn_featurizer.py to featurize images and store them in groups of 1000.
7.  In train_models.py, run:  train_svm_model() which will train and store your model.
8. Set the location of your model in city_app.py, host it somewhere, then run it, and you can be running your own City Tagger.

## <a name="credits">Credits
Thanks to the Visual Geometry Group, Oxford for designing the VGG-s net I used in my project.

VGG_CNN_S, model from the paper:
  "Return of the Devil in the Details: Delving Deep into Convolutional Nets"

Original source: https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9
License: non-commercial use only

## <a name="license">License

Non-commercial use only.  
