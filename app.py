# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:
# This script sets up a Dash web application for monitoring tool conditions and visual inspections in real-time.
# The application displays gauge charts, feature importance visualizations, system statistics, and terminal output. 
# The layout is built using Dash Bootstrap components to ensure a responsive design and user-friendly interface. 
# An interval component is included to periodically update the displayed data.

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from callbacks import update_gauge_charts

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Tool Condition and Visual Inspection Monitoring", className='mb-2', style={'textAlign': 'center'}),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='gauge-chart-1', config={"displayModeBar": False})], width=6),
        dbc.Col([dcc.Graph(id='feature-importance-chart', config={"displayModeBar": False})], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(id='gauge-chart-2', config={"displayModeBar": False})], width=6),
        dbc.Col([
            html.H3("System Stats"),
            dbc.Table(id="system-stats-table", bordered=True, striped=True)
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id='gauge-chart-3', config={"displayModeBar": False})], width=6),
        dbc.Col([
            html.H3("Terminal Output"),
            html.Pre(id="terminal-output", style={"background-color": "#f8f9fa", "padding": "10px", "border": "1px solid #ccc"})
        ], width=6)
    ]),
    
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

# Register callbacks
update_gauge_charts(app)

if __name__ == "__main__":
    from simulate import main
    main(app)
