from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import imageresize
from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd
import pickle
from PIL import Image

UPLOAD_FOLDER = 'static/uploads'
MULTI_UPLOAD_FOLDER = 'static/multipleuploads'
MULTI_UPLOAD_FOLDERjpeg = 'static/multipleuploadsjpeg'

ALLOWED_EXTENSIONS = set(['tif'])



app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MULTI_UPLOAD_FOLDER'] = MULTI_UPLOAD_FOLDER
app.config['MULTI_UPLOAD_FOLDERjpeg'] = MULTI_UPLOAD_FOLDERjpeg

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def signuppg():
    if request.method == 'POST':
        email = request.form['email']
        pswrd = request.form['psw']
        pswrdrepeat = request.form['psw-repeat']
        if pswrd==pswrdrepeat:
            data = [[email, pswrd]]
            df = pd.DataFrame(data, columns=['Email', 'Password'])
            with open('signupdata.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
            return render_template("loginpage.html")
        else:
            error = "Passwords mismatch !"
            return render_template("signup.html", error = error)
    return render_template("signup.html")


@app.route('/homepage.html')
def homepg():
    if not session.get('logged_in'):

        return render_template('loginpage.html')
    return render_template("homepage.html")

@app.route('/loginpage.html', methods=['GET', 'POST'])
def loginpg():
    if request.method == 'POST':
        email = request.form['email']
        pswrd = request.form['psw']
        z = pd.read_csv("signupdata.csv")
        indx = -1
        for i in range(0, len(z)):
            if z.Email[i]==email:
                indx = i
        if indx == -1:
            error = "Invalid Username !"
            return render_template("loginpage.html", error=error)
        elif z.Password[indx] == pswrd:
            session['logged_in'] = True
            return render_template("homepage.html")
        else:
            error = "Invalid Username or Password !"
            return render_template("loginpage.html", error=error)
            
         
            

    return render_template("loginpage.html")


@app.route('/instructionspage.html')
def instructionspage():
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    return render_template("instructionspage.html")


@app.route('/multiupload.html', methods=['GET','POST'])
def mutli():
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    elif request.method == 'POST':
        if 'file[]' not in request.files:
            flash('SELECT A FILE !')
            return render_template('multiupload.html')
        uploaded_files = request.files.getlist("file[]")
        
        filenames = []
        filenamesjpeg = []
        for file in uploaded_files:
            if file.filename == '':
                flash('NO FILE SELECTED !')
                return redirect(request.url)
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['MULTI_UPLOAD_FOLDER'], filename))
                print(filename)
                splitted = filename.split('.')[0]
                splitted = splitted+".jpeg"
                path=os.path.join(app.config['MULTI_UPLOAD_FOLDER'], filename)
                
                outfile = os.path.join(app.config['MULTI_UPLOAD_FOLDERjpeg'], splitted)
                im = Image.open(path)
                out = im.convert("RGB")
                out.save(outfile, "JPEG", quality=90)
                filenames.append(filename)

                filenamesjpeg.append(splitted)
            else:
                flash('FILE SHOULD BE IN .tif format only !')
                return render_template('multiupload.html')

        imageresize.imgrsz('static\\multipleuploads\\', 96)
        imageresize.imgrsz('static\\multipleuploadsjpeg\\', 200)

        model = pickle.load( open( "save.p", "rb" ) )
        imageresize.analysing(1, "multipredictions.csv", model, 'static\\multipleuploads\\')
        
        result=pd.read_csv('multipredictions.csv')
        out = []
        description = []
        for i in range(0, len(result)):
            if result.label[i] == 0.0:
                out.append('non cancerous')
                description.append('the AI predicts that the cell image does not show cancer symptoms')
            if result.label[i] == 1.0:
                out.append('cancerous')
                description.append('the AI predicts that the cell image might be cancerous')

        return render_template('result.html', filenames = filenamesjpeg, result = result, len = len(filenamesjpeg), out=out ,description = description)
              
    return render_template('multiupload.html')

@app.route('/static/multipleuploads/<filename>')
def uploaded_file(filename):
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    return send_from_directory(app.config['MULTI_UPLOAD_FOLDER'],
                               filename)


@app.route('/aboutpage.html')
def about():
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    return render_template('aboutpage.html')


@app.route('/result.html')
def resultpg():
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    return render_template('result.html')    

@app.route('/contactpage.html')
def contactpg():
    if not session.get('logged_in'):
        return render_template('loginpage.html')

    return render_template('contactpage.html')
   
@app.route("/logout")
def logout():
    if not session.get('logged_in'):
        return render_template('loginpage.html')
    else:
        session['logged_in'] = False
    return redirect('loginpage.html')


if __name__ == '__main__':
    app.run(debug=True)

