from flask import Flask, render_template, request
from database_func_2 import *
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
from plotly_1 import *
from utils_2 import *

app_2 = Flask(__name__)
app_2.config["SECRET_KEY"] = 'testkey'
app_2.config['UPLOAD_FOLDER'] = 'static/files'

@app_2.route('/',methods=['GET',"POST"])
@app_2.route('/home',methods=['GET',"POST"])
def home():
    return render_template('index_2.html')

@app_2.route('/submit', methods=['POST'])
def submit():
    userid = request.form['user_id']
    data = get_csv(userid)
    fig = generate_figure(data)
    fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    X_train, y_train, X_test, y_test, scaled_array,target_scaler = get_train_test(data)
    bayesian_lstm, criterion, optimizer = create_model(X_train, y_train, X_test, y_test, scaled_array)
    bayesian_lstm, train_loss = train_model(X_train, y_train,bayesian_lstm, criterion, optimizer)
    evaluation, testing_df, testing_truth_df = eval(data, bayesian_lstm, X_train, X_test,target_scaler) 
    fig_2 = generate_pred_figure(evaluation)
    fig_2_json = json.dumps(fig_2, cls=PlotlyJSONEncoder)
    test_uncertainty_df = get_test_uncertainty_df(testing_df, bayesian_lstm, X_test, target_scaler)
    fig_3, test_uncertainty_plot_df, truth_uncertainty_plot_df = generate_uncertainity_figure(test_uncertainty_df, testing_truth_df)
    return_statement = get_return(test_uncertainty_plot_df, truth_uncertainty_plot_df)
    print(return_statement)
    return render_template('result_2.html', fig_json=fig_json, fig_json_2 = fig_2_json, fig_json_3=fig_3.to_html(full_html=False, div_id='plot-3'), rs=return_statement)


if __name__=="__main__":
    app_2.run(port=5001, debug=True)

