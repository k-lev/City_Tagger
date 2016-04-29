from boto.s3.connection import S3Connection
import json
import io
import skimage.transform
import matplotlib.pyplot as plt
import urllib
import numpy as np
from lasagne.utils import floatX

def prep_image(url, mean_image):
    '''
    Input: Take url of image (Typically on s3, but can be any url)
    Take a url of an image.
    Resize image so the smallest dimension (h or w) is 256.
    Center crop the largest dimension 256 pixels.
    Output: 256x256x3 image
    '''
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    if len(im.shape) < 3:
        im = np.array((im,im,im))
        print im.shape
        im = np.swapaxes(im,0,1)
        im = np.swapaxes(im,1,2)
        print 'new shape: ',im.shape
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - mean_image
    return rawim, floatX(im[np.newaxis])
def get_aws_access():
	'''
	Input: None
	Output: String, String
	Read in the .json file where the aws access key and secret access key are stored.
	Output the access and secret_access_key.
	'''

	with open('/home/ubuntu/Projects/aws.json') as f:
		data = json.load(f)
		access_key = data['access-key']
		secret_access_key = data['secret-access-key']

	return access_key, secret_access_key

def print_bucket_files(bucket):
    '''
    Takes an s3 bucket address as input.
    Prints all files in the bucket.
    '''
    for i, key in enumerate(bucket.list()):
        print key.name.encode('utf-8')
        if i > 10:
            break

def get_bucket_files(bucket):
    '''
    Takes an s3 bucket address as input.
    Returns a list of all files in the bucket.
    '''
    chicago_list = []
    london_list = []
    sanfrancisco_list = []
    for key in bucket.list():
        filename = key.name.encode('utf-8')
        if filename[:7] == 'chicago':
            chicago_list.append(filename)
        elif filename[:6] == 'london':
            london_list.append(filename)
        else:
            sanfrancisco_list.append(filename)

    return chicago_list, london_list, sanfrancisco_list

def get_image_lists():
    '''
    Returns three lists of image locations:  chicago_list, london_list, sanfrancisco_list
    '''

    access_key, secret_access_key = get_aws_access()
    conn = S3Connection(access_key,secret_access_key)
    bucket = conn.get_bucket('rawcityimages')
    chicago_list, london_list, sanfrancisco_list = get_bucket_files(bucket)
    return chicago_list, london_list, sanfrancisco_list


def write_to_s3(bucket_name, filename):
    # get the keys
    access_key, secret_access_key = get_aws_access()

    #get a connection
    conn = S3Connection(access_key, secret_access_key)

    #get the bucket on s3
    bucket = conn.get_bucket(bucket_name)

    #save file to s3
    key = bucket.new_key(filename)
    key.set_contents_from_filename(filename)


if __name__ == '__main__':
    access_key, secret_access_key = get_aws_access()
    conn = S3Connection(access_key,secret_access_key)
    bucket = conn.get_bucket('rawcityimages')
    chicago_list, london_list, sanfrancisco_list = get_bucket_files(bucket)
    print "Chi %d", len(chicago_list)
    print "Lon %d", len(london_list)
    print "SF %d", len(sanfrancisco_list)
