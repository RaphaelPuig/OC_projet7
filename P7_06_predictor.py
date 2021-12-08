# Dash environment
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table

# application's pages handling and debug mode
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# data handling
import pandas as pd
import base64
import io

# modeling
import lightgbm
import pickle

external_stylesheets = ['assets//bWLwgP.css']

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server


#################### PREDICTOR (new clients from a uploaded file) ####################
# 1) load modeling utilities (trained model, imputer and scaler made on training base)
# 2) define predictor function called in parse_content function
# 3) define parse_content function (with display options)
# 4) app's layout with uploading data solution and result of parse_content function
# 5) callback with data retrieving

# Load model, imputer and scaler from file
loaded_model = pickle.load(open('assets//payback_predictor.pickle.dat', 'rb'))
loaded_imputer = pickle.load(open('assets//imputer.pickle.dat', 'rb'))
loaded_scaler = pickle.load(open('assets//scaler.pickle.dat', 'rb'))

def predictor(model, imputer, scaler, row):
    # Drop unwanted columns
    row_reduce = row.drop(columns=['SK_ID_CURR','TARGET'])
    # Impute missing value and normalize
    row_reduce = imputer.transform(row_reduce)
    row_reduce = scaler.transform(row_reduce)
    # Perform prediction
    pred_value = model.predict_proba(row_reduce)[0,1]
    return pred_value

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file. Please upload a csv or xls format file.'
        ])

    # Perform predictions on uploaded database
    pred=list()
    for i in df.index:
        pred.append(predictor(loaded_model, loaded_imputer, loaded_scaler, df.iloc[[i]]))

    # Selection of main features for final display
    df_extract = df[['SK_ID_CURR','EXT_SOURCE_3','EXT_SOURCE_2',
                     'EXT_SOURCE_1','DAYS_BIRTH','AMT_CREDIT']].copy()
                                                         
    # Convert 'DAYS' features in years
    df_extract['DAYS_BIRTH'] = df_extract['DAYS_BIRTH'].div(-365)

    # Rename columns (with units)
    df_extract.rename(columns={'DAYS_BIRTH': 'AGE (years)'}, inplace=True)
    df_extract.rename(columns={'AMT_CREDIT': 'AMT_CREDIT (dollars)'}, inplace=True)

    # Add the predictions to the dataframe
    df_extract['PREDICTION'] = pred

    # Round data to improve readabilty
    df_extract = df_extract.round({'EXT_SOURCE_3':3,
                                   'EXT_SOURCE_2':3, 
                                   'EXT_SOURCE_1':3,
                                   'AGE (years)':0,
                                   'AMT_CREDIT (dollars)':0,
                                   'PREDICTION': 3})

    return html.Div([
        html.Br(),
        html.Label('Main features and prediction of payback failure probability:'),
        html.Br(),
        dash_table.DataTable(
            data=df_extract.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df_extract.columns],
            sort_action='native',
            editable=False,
            style_data_conditional=[
                {
                    'if': {'filter_query': '{PREDICTION} < 0.33'},
                    'backgroundColor': 'rgb(220, 255, 220)',
                },
                {
                    'if': {'filter_query': '0.33 <= {PREDICTION} < 0.66'},
                    'backgroundColor': 'rgb(255, 245, 220)',
                },
                {
                    'if': {'filter_query': '{PREDICTION} >= 0.66'},
                    'backgroundColor': 'rgb(255, 220, 220)',
                },
                {
                    'if': {'state': 'active'},
                    'backgroundColor': 'rgb(220, 255, 255)',
                    'border': '1px solid rgb(210, 210, 210)'
                },
            ],
        ),
    ])

app.layout = html.Div([
    dcc.Upload(
        id='upload-data', 
        children = html.Button('Upload File'),
        multiple=True # Allow multiple files to be uploaded (strictly necessary!)
        ),
    html.Div(id='output-data-upload'),
])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True) # set True if development mode, False if production mode