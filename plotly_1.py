import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import pandas as pd
import json

def generate_figure(df):
    plot_length = 150
    plot_df = df.copy(deep=True).iloc[:plot_length]
    plot_df['weekday'] = plot_df['date'].dt.day_name()

    fig = px.line(plot_df,
                x="date",
                y="Energy_consumption", 
                color="weekday",
                )
    fig.update_layout(plot_bgcolor = 'rgba(217, 217, 217, 1)', 
                      paper_bgcolor='rgba(210, 210, 210, 1)',
                      width=1300)
    return pio.to_json(fig)

def generate_pred_figure(evaluation):
    fig = px.line(evaluation.loc[evaluation['date'].between('2014-04-14', '2016-10-30')],
                 x="date",
                 y="Energy_consumption",
                 color="source"
                 )
    
    fig.update_layout(plot_bgcolor = 'rgba(217, 217, 217, 1)', 
                      paper_bgcolor='rgba(210, 210, 210, 1)',
                      width=1300,
                      legend=dict(
                                font=dict(
                                    size=10
                                )),
                            
                    )
    
    return pio.to_json(fig)

def generate_uncertainity_figure(test_uncertainty_df, testing_truth_df):
    test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
    test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
    truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

    # Create the upper bound trace
    upper_trace = go.Scatter(
        x=test_uncertainty_plot_df['date'],
        y=test_uncertainty_plot_df['upper_bound'],
        mode='lines',
        fill=None,
        name='99% Upper CB'
    )

    # Create the lower bound trace
    lower_trace = go.Scatter(
        x=test_uncertainty_plot_df['date'],
        y=test_uncertainty_plot_df['lower_bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 211, 0, 0.1)',
        name='99% Lower CB'
    )

    # Create the real values trace
    real_trace = go.Scatter(
        x=truth_uncertainty_plot_df['date'],
        y=truth_uncertainty_plot_df['Energy_consumption'],
        mode='lines',
        fill=None,
        name='Real Values'
    )

    # Create the data list with all the traces
    data = [upper_trace, lower_trace, real_trace]

    # Create the figure object
    fig = go.Figure(data=data)
    fig.update_layout(plot_bgcolor = 'rgba(217, 217, 217, 1)', 
                      paper_bgcolor='rgba(210, 210, 210, 1)',
                      width=1310,
                      height=500)

    return fig, test_uncertainty_plot_df, truth_uncertainty_plot_df


