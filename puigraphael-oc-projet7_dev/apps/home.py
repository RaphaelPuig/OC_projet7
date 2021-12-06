# Dash environment
import dash_core_components as dcc
import dash_html_components as html


layout = html.Div([
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Welcome to HomeCredit application', style={'textAlign': 'center'}),
    html.Br(),
    html.H3([    
        dcc.Link('Dashboard', href='/dashboard/'),
        ]),
    html.H6('Displays the main features and the payback failure probability for the clients already in the database.'),
    html.Br(),
    html.H3([
        dcc.Link('Predictor', href='/predictor/'),
        ]),
    html.H6('Displays the main features and the payback failure probability for the clients from an uploaded database.'),
    html.H6('Suppose a database in csv or xls format to load.'),
    html.Br(),
    html.Br(),
    html.H6('Click on the appropriate title to make your choice.')
    ])