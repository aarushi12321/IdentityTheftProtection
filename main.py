from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
import os
import torch
from wtforms.validators import InputRequired
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
from utils import preprocess_json_data, turn_compatible, reshape_to_deep, self_define_cnn_kernel_process
from database_func import add_user, add_flag, get_result
import webbrowser
#from index_dash import dash_index as dash_index

# Instances
app = Flask(__name__)
app.config["SECRET_KEY"] = 'testkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
model = pickle.load(open("model1.pkl", "rb"))

# Classes
class UploadFileForm(FlaskForm):
    file=FileField("File", validators=[InputRequired()])
    submit=SubmitField("Upload File")
    
# Routes and functions
@app.route('/',methods=['GET',"POST"])
@app.route('/home',methods=['GET',"POST"])
def home():
    form=UploadFileForm()
    if form.validate_on_submit():
        file=form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        rel_path=os.path.relpath(file_path,app.root_path)
        return extract_data(rel_path)
    return render_template('index.html', form=form)

@app.route('/extract/<filename>')
def extract_data(filename):
    with open(filename,'r') as f:
        data=json.load(f)
        userid = data["userid"]
        flag = add_user(data)
        if flag == 0:
            result, output = predict(data)
            add_flag(userid, output)
            # instruct to re-calculate variables in dash app
        else:
            n_result = get_result(userid)
            if int(n_result[0]) == 1:
                result = 'Malicious user data'
            else:
                result = 'Beneign user data'

    if flag == 0:
        fl_val = "New user added to the database."
    else:
        fl_val = "User already exists in the database."

    return render_template('result.html',result=result, userid=userid, fl_val=fl_val)


@app.route('/dash_app')
def dash_app():
    return redirect('http://127.0.0.1:8050/', code=302)

@app.route('/go-to-app-2')
def go_to_app_2():
    return redirect('http://127.0.0.1:5001/')

def predict(json_features):
    features = preprocess_json_data(json_features)
    features = np.array(features)

    x_wide = np.reshape(features, (1, features.shape[0]))
    x_wide = turn_compatible(x_wide)
    x_deep = reshape_to_deep(x_wide)
    x_pre = self_define_cnn_kernel_process(x_deep)

    prediction = model.predict([x_wide, x_pre])
    
    if int(prediction) == 1:
        prediction_output = 'Malicious user data'

    else:
        prediction_output = 'Beneign user data'
    return prediction_output, int(prediction)

if __name__=="__main__":
    app.run(port=5000, debug=True)