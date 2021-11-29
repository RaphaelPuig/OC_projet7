# Dash environment
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

# plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# data handling
import pandas as pd
import numpy as np
import scipy
from scipy import stats

external_stylesheets = ['assets//bWLwgP.css']

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

################## DASHBOARD (clients of the database) ##################
# 1) compute all KDE plots
# 2) app's layout with main features values table and created plots
# 3) callback with main features values retrieving and creation of plots

### load database from data folder ###
train_df = pd.read_csv('assets//train_extract.csv', dtype=np.float32) # for KDE plots
global_df = pd.read_csv('assets//global_general_extract.csv', dtype=np.float32) # clients data

available_clients = global_df['SK_ID_CURR']

### EXT_SOURCE_1 KDE
xmin_ES1 = train_df['EXT_SOURCE_1'].min()
xmax_ES1 = train_df['EXT_SOURCE_1'].max()
X_ES1 = np.linspace(xmin_ES1, xmax_ES1, 100)

values_ES1_0 = train_df[train_df['TARGET']==0]['EXT_SOURCE_1'].dropna()
values_ES1_1 = train_df[train_df['TARGET']==1]['EXT_SOURCE_1'].dropna()

kernel_ES1_0 = stats.gaussian_kde(values_ES1_0, bw_method='scott')
kernel_ES1_1 = stats.gaussian_kde(values_ES1_1, bw_method='scott')

Z_ES1_0 = np.reshape(kernel_ES1_0(X_ES1).T, X_ES1.shape)
Z_ES1_1 = np.reshape(kernel_ES1_1(X_ES1).T, X_ES1.shape)

### EXT_SOURCE_2 KDE
xmin_ES2 = train_df['EXT_SOURCE_2'].min()
xmax_ES2 = train_df['EXT_SOURCE_2'].max()
X_ES2 = np.linspace(xmin_ES2, xmax_ES2, 100)

values_ES2_0 = train_df[train_df['TARGET']==0]['EXT_SOURCE_2'].dropna()
values_ES2_1 = train_df[train_df['TARGET']==1]['EXT_SOURCE_2'].dropna()

kernel_ES2_0 = stats.gaussian_kde(values_ES2_0, bw_method='scott')
kernel_ES2_1 = stats.gaussian_kde(values_ES2_1, bw_method='scott')

Z_ES2_0 = np.reshape(kernel_ES2_0(X_ES2).T, X_ES2.shape)
Z_ES2_1 = np.reshape(kernel_ES2_1(X_ES2).T, X_ES2.shape)

### EXT_SOURCE_3 KDE
xmin_ES3 = train_df['EXT_SOURCE_3'].min()
xmax_ES3 = train_df['EXT_SOURCE_3'].max()
X_ES3 = np.linspace(xmin_ES3, xmax_ES3, 100)

values_ES3_0 = train_df[train_df['TARGET']==0]['EXT_SOURCE_3'].dropna()
values_ES3_1 = train_df[train_df['TARGET']==1]['EXT_SOURCE_3'].dropna()

kernel_ES3_0 = stats.gaussian_kde(values_ES3_0, bw_method='scott')
kernel_ES3_1 = stats.gaussian_kde(values_ES3_1, bw_method='scott')

Z_ES3_0 = np.reshape(kernel_ES3_0(X_ES3).T, X_ES3.shape)
Z_ES3_1 = np.reshape(kernel_ES3_1(X_ES3).T, X_ES3.shape)

### AMT_CREDIT KDE
xmin_AMT = train_df['AMT_CREDIT'].min()
xmax_AMT = train_df['AMT_CREDIT'].max()
X_AMT = np.linspace(xmin_AMT, xmax_AMT, 100)

values_AMT_0 = train_df[train_df['TARGET']==0]['AMT_CREDIT'].dropna()
values_AMT_1 = train_df[train_df['TARGET']==1]['AMT_CREDIT'].dropna()

kernel_AMT_0 = stats.gaussian_kde(values_AMT_0, bw_method='scott')
kernel_AMT_1 = stats.gaussian_kde(values_AMT_1, bw_method='scott')

Z_AMT_0 = np.reshape(kernel_AMT_0(X_AMT).T, X_AMT.shape)
Z_AMT_1 = np.reshape(kernel_AMT_1(X_AMT).T, X_AMT.shape)

### Age KDE
xmin_age = (train_df['DAYS_BIRTH']/-365).min()
xmax_age = (train_df['DAYS_BIRTH']/-365).max()
X_age = np.linspace(xmin_age, xmax_age, 100)

values_age_0 = (train_df[train_df['TARGET']==0]['DAYS_BIRTH']/-365).dropna()
values_age_1 = (train_df[train_df['TARGET']==1]['DAYS_BIRTH']/-365).dropna()

kernel_age_0 = stats.gaussian_kde(values_age_0, bw_method='scott')
kernel_age_1 = stats.gaussian_kde(values_age_1, bw_method='scott')

Z_age_0 = np.reshape(kernel_age_0(X_age).T, X_age.shape)
Z_age_1 = np.reshape(kernel_age_1(X_age).T, X_age.shape)

### predictions KDE
xmin_pred = global_df['PREDICTION'].min()
xmax_pred = global_df['PREDICTION'].max()
X_pred = np.linspace(xmin_pred, xmax_pred, 100)

values_pred = global_df['PREDICTION']

kernel_pred = stats.gaussian_kde(values_pred, bw_method='scott')

Z_pred = np.reshape(kernel_pred(X_pred).T, X_pred.shape)

app.layout = html.Div([
    html.Label('Client'),
    dcc.Dropdown(
        id='SK_ID_CURR',
        options=[{'label': i, 'value': i} for i in available_clients],
        value=np.min(available_clients)
        ),
    
    html.Table([
        html.Tr([html.Td(['EXT_SOURCE_1:']), html.Td(id='ES1'), 
                 html.Td(['EXT_SOURCE_2:']), html.Td(id='ES2'),
                 html.Td(['EXT_SOURCE_3:']), html.Td(id='ES3'),
                 html.Td(['AMT_CREDIT:']), html.Td(id='AMT'),
                 html.Td(['Age:']), html.Td(id='Age'),
                 html.Td(['Payback failure probability:']), html.Td(id='pred')
                 ]),
    ]),
    
    html.Div(
        dcc.Graph(id='graphic', config={'responsive': True})
        ),
])

@app.callback(
    Output('graphic','figure'),
    Output('ES1','children'),
    Output('ES2','children'),
    Output('ES3','children'),
    Output('AMT','children'),
    Output('Age', 'children'),
    Output('pred','children'),
    Input('SK_ID_CURR', 'value'))
def update_figure(SK_ID_CURR):
    pred = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].PREDICTION.item(), decimals=3)
    ES1_value = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].EXT_SOURCE_1.item(), decimals=3)
    ES2_value = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].EXT_SOURCE_2.item(), decimals=3)
    ES3_value = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].EXT_SOURCE_3.item(), decimals=3)
    AMT_value = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].AMT_CREDIT.item(), decimals=3)
    age_value = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].DAYS_BIRTH.item()/-365, decimals=0)
    pred = np.round(global_df[global_df.SK_ID_CURR == SK_ID_CURR].PREDICTION.item(), decimals=3)

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=('EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                                        'AMT_CREDIT', 'Age', 'Payback failure probability'))
    # EXT_SOURCE_1
    fig.add_trace(go.Scatter(x=X_ES1, y=Z_ES1_0, marker = {'color' : 'blue'}, name='success'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_ES1, y=Z_ES1_1, marker = {'color' : 'red'},  name='failure'), row=1, col=1)

    # EXT_SOURCE_2
    fig.add_trace(go.Scatter(x=X_ES2, y=Z_ES2_0, marker = {'color' : 'blue'}, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_ES2, y=Z_ES2_1, marker = {'color' : 'red'}, showlegend=False), row=1, col=2)

    # EXT_SOURCE_3
    fig.add_trace(go.Scatter(x=X_ES3, y=Z_ES3_0, marker = {'color' : 'blue'}, showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=X_ES3, y=Z_ES3_1, marker = {'color' : 'red'}, showlegend=False), row=1, col=3)

    # AMT_CREDIT
    fig.add_trace(go.Scatter(x=X_AMT, y=Z_AMT_0, marker = {'color' : 'blue'}, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=X_AMT, y=Z_AMT_1, marker = {'color' : 'red'}, showlegend=False), row=2, col=1)

    # Age
    fig.add_trace(go.Scatter(x=X_age, y=Z_age_0, marker = {'color' : 'blue'}, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=X_age, y=Z_age_1, marker = {'color' : 'red'}, showlegend=False), row=2, col=2)

    # Predictions    
    fig.add_trace(go.Scatter(x=X_pred, y=Z_pred, marker = {'color' : 'purple'}, showlegend=False), row=2, col=3)

    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.1,
                                  xanchor="right",
                                  x=1)
                      )

    fig.update_layout(title={'text': 'Kernel density estimators of the main features and the payback failure probability'})

    # Vertical lines
    if np.isnan(ES1_value)==False:
        fig.add_trace(go.Scatter(x=[ES1_value, ES1_value], y=[0,np.max([Z_ES1_0,Z_ES1_1])], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 showlegend=False),
                                 row=1, col=1)

    if np.isnan(ES2_value)==False:
        fig.add_trace(go.Scatter(x=[ES2_value, ES2_value], y=[0,np.max([Z_ES2_0,Z_ES2_1])], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 showlegend=False),
                                 row=1, col=2)

    if np.isnan(ES3_value)==False:
        fig.add_trace(go.Scatter(x=[ES3_value, ES3_value], y=[0,np.max([Z_ES3_0,Z_ES3_1])], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 showlegend=False),
                                 row=1, col=3)

    if np.isnan(AMT_value)==False:
        fig.add_trace(go.Scatter(x=[AMT_value, AMT_value], y=[0,np.max([Z_AMT_0,Z_AMT_1])], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 showlegend=False),
                                 row=2, col=1)
    

    if np.isnan(age_value)==False:
        fig.add_trace(go.Scatter(x=[age_value, age_value], y=[0,np.max([Z_age_0,Z_age_1])], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 showlegend=False),
                                 row=2, col=2)

    if np.isnan(pred)==False:
        fig.add_trace(go.Scatter(x=[pred, pred], y=[0,np.max(Z_pred)], mode='lines',
                                 line={'color': 'green', 'width': 1, 'dash': 'dot'},
                                 name='client\'s value'),
                                 row=2, col=3)

    fig.update_yaxes(title='density')

    return fig, ES1_value, ES2_value, ES3_value, AMT_value, age_value, pred #empl_value, pred


if __name__ == '__main__':
    app.run_server(debug=True) # set True if development mode, False if production mode