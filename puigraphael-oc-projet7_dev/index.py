#import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# connect to application pages
from apps import home, dashboard, predictor

# connect to main app.py file
from app import app
from app import server


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[])
])


@app.callback(Output(component_id='page-content', component_property='children'),
              [Input(component_id='url', component_property='pathname')])
def display_page(pathname):
    if pathname == '/dashboard/':
        return dashboard.layout
    if pathname == '/predictor/':
        return predictor.layout
    else:
        return home.layout


if __name__ == '__main__':
    app.run_server(debug=False)
