import pandas as pd
from dash_utils import total_consumption, get_users, get_dataframe
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from current_time import get_time
import dash_bootstrap_components as dbc
from forecast import app_2 as app_2, home as flask_home_forecast
from flask import Flask, redirect, url_for
import requests

dash_app = dash.Dash(__name__)

global combined_df, mal_df, ben_df

combined_df = get_dataframe("SELECT * FROM combined_data")
mal_df = get_dataframe("SELECT * FROM malicious_users")
ben_df = get_dataframe("SELECT * FROM benign_users")

def regen_df():
    combined_df = get_dataframe("SELECT * FROM combined_data")
    mal_df = get_dataframe("SELECT * FROM malicious_users")
    ben_df = get_dataframe("SELECT * FROM benign_users")

def get_data(df, sum=False):
    new_df = df.iloc[:,1:]
    new_df = new_df.iloc[:,:-1]

    df_2 = pd.DataFrame()
    if sum == True:
        df_2['date'] = new_df.sum().index
        df_2['energy'] = new_df.sum().values
    
    else:
        df_2['date'] = new_df.columns
        df_2['energy'] = new_df.values[0]

    return df_2

# This is the main container
dash_app.layout = html.Div( className="Main",
    children= [
    
    dbc.Row(children= [
        html.Div( className="Top-panel",
        children= [
            html.Ul( className="Top-panel-list",
            children= [
                html.Li(html.P('Dashboard'), id="heading")
            ]
        )
    ]
    )]
    ),

    dbc.Row(children = [
        dbc.Col(
            # This is container for last update
            html.Div( className = "last-update",id='last-update-id',
            children = [
                html.P('Last Updated: ' + get_time())
            ]
        )
        ) 
    ]
    ),

    dbc.Row( className= "info-class", id="info-class-id",
        children= [

        dbc.Col(
            # This is container for total users
            html.Div( className="info-tab total",
            children = [
                html.H3('Total Users '),
                html.P(get_users(combined_df))
            ]
        )
        ),
        
        dbc.Col( 
            # This is container for benign users
            html.Div( className="info-tab benign",
            children = [
                html.H3('Benign Users '),
                html.P(get_users(ben_df))
            ]
        ) 
        ),

        dbc.Col(
            # This is container for Malicious users
            html.Div( className="info-tab malicious",
            children = [
                html.H3('Malicious Users '),
                html.P(get_users(mal_df))
            ]
        )
        )
        ]
    ),

    #----------------------------------------------------------------------------------------------------------------------
    html.Div(className="graph-buttons",
    children=[
        html.Ul(
    children=[
    html.Li(html.Button('Total Energy Consumption', id='button-1', n_clicks=0)),
    html.Li(html.Button("Benign Users' Consumption", id='button-2', n_clicks=0)),
    html.Li(html.Button("Malicious Users' Consumption", id='button-3', n_clicks=0)),
    html.Li(html.Div(className="rev-class", id='button-4',
                children = [
                    dcc.Input(id='user-id-input', placeholder='Enter user id', type='text'),
                    html.A(children=[html.Button('Give user insights', id='retrieve-data-button',)], 
                           href="#output-container")


    ]
    ))
    ]
        )
    ]),
    
    # Create the output div for the graphs
    html.Div(id='graph-output',
             children=[
                # Default graph (Graph 1)
                html.Div(className='graph', id='graph', children=[
                    dcc.Graph(id='total-energy-graph', figure=px.line(get_data(combined_df, sum=True), x='date', y='energy').update_layout(
                                #title="Total users' Consumption",
                                xaxis_title='Date',
                                yaxis_title='Total Energy Consumption',
                                plot_bgcolor = 'rgba(217, 217, 217, 1)',
                                paper_bgcolor='rgba(210, 210, 210, 1)'
                                ))
                ]),
        
                # Graph 2 (hidden by default)
                html.Div(className='graph', id='graph-2', style={'display': 'none'}, children=[
                    dcc.Graph(id='ben-energy-graph', figure=px.line(get_data(ben_df, sum=True), x='date', y='energy').update_layout(
                                #title="Total users' Consumption",
                                xaxis_title='Date',
                                yaxis_title="Benign Users' Consumption",
                                plot_bgcolor = 'rgba(217, 217, 217, 1)',
                                paper_bgcolor='rgba(210, 210, 210, 1)'
                                ))
                ]),
        
                # Graph 3 (hidden by default)
                html.Div(className='graph', id='graph-3', style={'display': 'none'}, children=[
                    dcc.Graph(id='mal-energy-graph', figure=px.line(get_data(mal_df, sum=True), x='date', y='energy').update_layout(
                                #title="Total users' Consumption",
                                xaxis_title='Date',
                                yaxis_title="Malicious Users' Consumption",
                                plot_bgcolor = 'rgba(217, 217, 217, 1)',
                                paper_bgcolor='rgba(210, 210, 210, 1)'
                                ))
                ])            
             ]),

    html.Div(id='output-container'),
    html.H1(id='df-output'),
    dcc.Interval(id='interval-component',
            interval=60*1000, # in milliseconds
            n_intervals=0),
    #----------------------------------------------------------------------------------------------------------------------
])

@dash_app.callback(
    Output('output-container', 'children'),  
    [Input('retrieve-data-button', 'n_clicks')],  
    [dash.dependencies.State('user-id-input', 'value')] 
)
def retrieve_data(n_clicks, user_id):
    if n_clicks is not None and n_clicks > 0:  
        if user_id is not None: 
            retrieved_data = combined_df[combined_df.userid == user_id]
            if not retrieved_data.empty:
                total_con = total_consumption(retrieved_data)
                flag = retrieved_data.loc[:,'FLAG'].values

                if flag[0] == 0:
                    flag ='Benign'
                else:
                    flag = 'Malicious'

                return html.Div(className = "user-div", id="redirect-div", children=[
                                        html.P('User id: {}'.format(user_id)),
                                        dcc.Graph(id='user-eneergy-graph', figure = px.line(get_data(retrieved_data),
                                                                                                x='date',
                                                                                                y='energy').update_layout(
                            xaxis_title='Date',
                            yaxis_title='Daily Energy Consumption',
                            plot_bgcolor = 'rgba(217, 217, 217, 1)',
                            paper_bgcolor='rgba(210, 210, 210, 1)')),
                            html.P('Total Energy consumption: {}'.format(total_con)),
                            html.P('User type: {}'.format(flag)),
                                          ])
        
            else:
                return html.Div('No data found for User ID: {}'.format(user_id))
        else:
            return html.P("Please enter a User ID.")
    else:
        return html.Div()
    
@dash_app.callback(
    [Output('graph', 'style'), Output('graph-2', 'style'), Output('graph-3', 'style')],
    [Input('button-1', 'n_clicks'), Input('button-2', 'n_clicks'), Input('button-3', 'n_clicks')],
    [State('graph', 'style'), State('graph-2', 'style'), State('graph-3', 'style')])

def generate_graph(n_clicks_1, n_clicks_2, n_clicks_3, style_1, style_2, style_3):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Set the style for the graph that was clicked
    if button_id == 'button-1':
        return {}, {'display': 'none'}, {'display': 'none'}
    elif button_id == 'button-2':
        return {'display': 'none'}, {}, {'display': 'none'}
    elif button_id == 'button-3':
        return {'display': 'none'}, {'display': 'none'}, {}
    else:
        return {'display': 'none'}, {}, {'display': 'none'}

@dash_app.callback(Output('info-class-id','children'),Input('interval-component','n_intervals'))
def update_metrics(n):
    combined_df = get_dataframe("SELECT * FROM combined_data")
    mal_df = get_dataframe("SELECT * FROM malicious_users")
    ben_df = get_dataframe("SELECT * FROM benign_users")
    return [

        dbc.Col(
            # This is container for total users
            html.Div( className="info-tab total",
            children = [
                html.H3('Total Users '),
                html.P(get_users(combined_df))
            ]
        )
        ),
        
        dbc.Col( 
            # This is container for benign users
            html.Div( className="info-tab benign",
            children = [
                html.H3('Benign Users '),
                html.P(get_users(ben_df))
            ]
        ) 
        ),

        dbc.Col(
            # This is container for Malicious users
            html.Div( className="info-tab malicious",
            children = [
                html.H3('Malicious Users '),
                html.P(get_users(mal_df))
            ]
        )
        )
        ]

@dash_app.callback(Output('last-update-id','children'),Input('interval-component','n_intervals'))
def update_last_updated(n):
    return [
        html.P('Last Updated: ' + get_time())
    ]

@dash_app.callback(Output('df-output', 'children'),Input('interval-component', 'n_intervals'))
def update_output_df(n_intervals):
    # Call the function inside the callback function
    return regen_df()

if __name__ == '__main__':
    dash_app.run_server(debug=True)