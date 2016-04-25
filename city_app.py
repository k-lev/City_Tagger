import os
from flask import Flask, request, render_template, send_file, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import urllib2
from os.path import basename
import urlparse

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', '.bmp'])

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

# home page
@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html', title='U-Bendare, your automatic city identifier!')

@app.route('/predict_file', methods=['GET', 'POST'])
def predict_file():
    file = request.files['filename']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        full_path = 'localhost:8080/'+full_path
        # features = featurize_image(filename)
        # prediction = model.predict(X)

    else:
        return render_template('predict_file.html', title='Bad file!')
    return render_template('predict_file.html', title='U-Bendare has located your city!', filename=filename, prediction='Chicago')

@app.route('/predict_link', methods=['POST'])
def predict_link():
    imgUrl = request.form['link_address']
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
        # features = featurize_image(filename)
        # prediction = model.predict(X)
    else:
        return render_template('predict_file.html', title='Bad link!')
    return render_template('predict_file.html', title='Bendare has located your city!', filename=filename, prediction='Chicago')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['filename']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file',
                                filename=filename))

@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
