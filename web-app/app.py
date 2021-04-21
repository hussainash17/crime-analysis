from flask import Flask,flash, request, render_template, jsonify, redirect
from flask import url_for, send_from_directory
from forms import ContactForm
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__,static_folder='static')
app.secret_key = 'dev fao football app'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, './static')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['csv', 'json', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are csv, json, png, jpg, jpeg, gif')
            return redirect(request.url)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/<path:filename>')  
def send_file(filename):  
    return send_from_directory(app.static_folder, filename)

@app.route('/location', methods=["GET","POST"])
def get_contact():
    form = ContactForm()
    if request.method == 'POST':
        country =  request.form["country"]
        region = request.form["region"]
        year = request.form["year"]
        latitude = request.form["latitude"]
        longitude = request.form["longitude"]
        targetType = request.form["targetType"]
        attackType = request.form["attackType"]
        weaponType = request.form["weaponType"]
        weaponSubType = request.form["weaponSubType"]
        suicide = request.form["suicide"]
        nkill = request.form["nkill"]
        nwonded = request.form["nwonded"]
        message = request.form["message"]
        res = pd.DataFrame({'country':country,'region':region, 'year': year, 'latitude':latitude, 'longitude':longitude,'targetType': targetType,'attackType': attackType, 'weaponType': weaponType ,'weaponSubType': weaponSubType,'suicide': suicide,'nkill':nkill,'nwonded': nwonded, 'message':message}, index=[0])
        res.to_csv('./userData.csv')
        return render_template('contact.html', form=form)
    else:
        return render_template('contact.html', form=form)


@app.route('/map')
def form():
    return render_template('map.html')

@app.route('/data', methods=["POST"])
def data():
    form_data = request.form
    print(form_data)
    return render_template('data.html', form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
