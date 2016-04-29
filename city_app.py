import os
from os.path import basename
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import urllib2
import urlparse
import urllib
from appcode.vgg_nn_featurizer import load_precreated_vgg_nn_and_mean_img
from appcode.process_images import prep_image
import cPickle as pickle
from nolearn.lasagne import NeuralNet
from sklearn.ensemble import RandomForestClassifier

#****************** GLOBALS *********************************
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/ubuntu/Projects/MyProject/data/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', '.bmp'])
CITY_LABELS = ['San Francisco','Chicago','London']
#*************************************************************

def allowed_file(filename):
    '''
    Input:  user selected filename.
    Output:  true if filename is in 'png', 'jpg', 'jpeg', 'gif' or '.bmp'
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def allowed_url(imgUrl):
    '''
    Input:  user's given url.
    Return true only if urls point directly to images.
    '''
    print imgUrl
    return imgUrl.lower().endswith('.jpeg') or \
        imgUrl.lower().endswith('.jpg') or \
        imgUrl.lower().endswith('.gif') or \
        imgUrl.lower().endswith('.png') or \
        imgUrl.lower().endswith('.bmp')

def predict_tag(filename):
    '''
    Imput: filename where image has been saved locally
    Take the image, run it though the neural net to featurize it.
    Run the features through the model to get predictions.
    Output:  list of probabilities for each category
             -- Category order:  [San Francisco, Chicago, London]
    '''

    # resize image
    raw_image, clean_image = prep_image(filename, MEAN_IMAGE)
    # feturize image via vgg neural net
    features = VGG_NN_FEATURIZER.predict_proba(clean_image)

    # Get the probability for each possible class (city)
    pred_probs = model.predict_proba(features)
    return pred_probs


# home page
@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html', title='U-Bendare, your automatic city identifier!')

@app.route('/predict_file', methods=['GET', 'POST'])
def predict_file():
    '''
    Retreive the user's file.
    Save it locally.
    Run the image through the model.
    Pass the details of the top two results to the predict_results.html template
    '''
    file = request.files['filename']
    # Ensure file has allowed image extensions.
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        full_path = full_path
        pred_probas = predict_tag(full_path)
        ordered_predictions = np.argsort(pred_probas[0])
        print ordered_predictions
        prediction = CITY_LABELS[ordered_predictions[-1]]
        prediction2 = CITY_LABELS[ordered_predictions[-2]]
        prob1 = pred_probas[0][ordered_predictions[-1]]
        prob2 = pred_probas[0][ordered_predictions[-2]]

    else:
        return render_template('predict_file.html', title='Bad file!')
    return render_template('predict_file.html', title='U-Bendare has located your city!', filename=filename,
                            prediction=prediction, prediction2=prediction2, prob1 = prob1,prob2 = prob2)


@app.route('/predict_link', methods=['POST'])
def predict_link():
    '''
    Retreive image from the user's input url.
    Save it locally.
    Run the image through the model.
    Pass the details of the top two results to the predict_results.html template
    '''
    imgUrl = request.form['link_address']
    # Ensure url has allowed image extensions.
    if imgUrl and allowed_url(imgUrl):
        try:
            print "here ********************** ",imgUrl
            imgData = urllib2.urlopen(imgUrl).read()
            filename = basename(urlparse.urlsplit(imgUrl)[2])
            output = open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'wb')
            output.write(imgData)
            output.close()
        except Exception, e:
            print str(e)
            return render_template('index.html')

        #  Get teh image; pass it tot he predictor
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        full_path = full_path
        pred_probas = predict_tag(full_path)
        ordered_predictions = np.argsort(pred_probas[0])
        prediction = CITY_LABELS[ordered_predictions[-1]]
        prediction2 = CITY_LABELS[ordered_predictions[-2]]
        prob1 = pred_probas[0][ordered_predictions[-1]]
        prob2 = pred_probas[0][ordered_predictions[-2]]
    else:
        return render_template('predict_file.html', title='Bad link!')
    return render_template('predict_file.html', title='U-Bendare has located your city!', filename=filename,
                            prediction=prediction, prediction2=prediction2, prob1 = prob1,prob2 = prob2)

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['filename']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)

        # Move the file form the temporal folder to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':

    ## *****************  Get pretrained vgg_nn and mean image data   #######################
    VGG_NN_FEATURIZER, MEAN_IMAGE = load_precreated_vgg_nn_and_mean_img()

    # load model
    with open("/data/tmp_rf_model.pkl") as f_un:
        model = pickle.load(f_un)


    app.run(host='0.0.0.0', port=8000, debug=True)
