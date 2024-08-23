from dash import  Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from charts import create_gauge_chart, create_feature_importance_chart
from utils import get_overall_top_10_features
import psutil
import time
from simulate import main
import global_vars
# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:
# This script defines the `update_gauge_charts` function for a Dash application.
# The function is responsible for updating the gauge charts, terminal outputs,
# feature importance charts, and system statistics table based on real-time datafrom machine learning models. 
# It utilizes global variables for model and feature information and integrates system resource statistics using the psutil library.

def update_gauge_charts(app):
    @app.callback(
        [Output(f'gauge-chart-{i}', 'figure') for i in range(1, 4)] +
        [Output("terminal-output", "children"),
         Output('feature-importance-chart', 'figure'),
         Output('system-stats-table', 'children')],
        Input('interval-component', 'n_intervals')
    )

    def update_gauge_charts(n_intervals):
        figures = []
        terminal_output = ""
        system_stats = []

        # Ensure the generator is initialized
        if global_vars.generator is None:
            raise RuntimeError("Generator is not initialized. Ensure that simulate.py's main() has been executed.")

        # Use the global variables directly
        for condition_value, inspection_value, status_message, label in global_vars.generator:
            index = global_vars.labels.index(label)
            figures.append(create_gauge_chart(condition_value, inspection_value, label))
            terminal_output += status_message + "\n"
            if len(figures) == len(global_vars.labels):
                break

        # Generate the feature importance chart
        overall_top_10_features = get_overall_top_10_features(
            [(global_vars.tc_features.columns[i], v) for i, v in enumerate(global_vars.tc_model.feature_importances_)],
            [(global_vars.vi_features.columns[i], v) for i, v in enumerate(global_vars.vi_model.feature_importances_)]
        )
        feature_importance_figure = create_feature_importance_chart(overall_top_10_features)
        
        # Generate system stats
        cpu_usage = psutil.cpu_percent(interval=0)
        memory_info = psutil.virtual_memory()
        estimated_power_usage = (cpu_usage / 100) * 45  # Adjust this value to your CPU's TDP

        system_stats.append(html.Tr([html.Td("CPU Usage"), html.Td(f"{cpu_usage}%")]))
        system_stats.append(html.Tr([html.Td("Memory Usage"), html.Td(f"{memory_info.percent}%")]))
        system_stats.append(html.Tr([html.Td("Estimated Power Usage"), html.Td(f"{estimated_power_usage:.2f} Watts")]))
        system_stats.append(html.Tr([html.Td("Timestamp"), html.Td(time.strftime('%Y-%m-%d %H:%M:%S'))]))

        return figures[0], figures[1], figures[2], terminal_output, feature_importance_figure, system_stats
