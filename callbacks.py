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
from system_stats import print_system_stats
   # Add global variables for tracking total confidence and count of predictions
total_power_used = 0.0
cpu_usage_sum = 0.0
memory_usage_sum = 0.0
interval_count = 0
# Add global variables for tracking total confidence and count of predictions
total_tc_confidence = 0.0
total_vi_confidence = 0.0
prediction_count = 0
CPU_TDP = 45  # Adjust this value to your CPU's TDP
   
def update_gauge_charts(app):
    @app.callback(
        [Output(f'gauge-chart-{i}', 'figure') for i in range(1, 4)] +
        [Output("terminal-output", "children"),
        Output('feature-importance-chart', 'figure'),
        Output('system-stats-table', 'children')],
        Input('interval-component', 'n_intervals')
    )
    def update_gauge_charts(n_intervals):
        global total_power_used, cpu_usage_sum, memory_usage_sum, interval_count
        global estimated_power_usage
        global tc_features, vi_features
        figures = []
        terminal_output = ""
        system_stats = []

        # Generate the gauge charts
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
        
    
        interval_count += 1  # Increment the count of intervals

        # Set interval to None to avoid unintended delays
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()

        # Calculate the estimated power usage
        estimated_power_usage = (cpu_usage / 100) * global_vars.CPU_TDP  # in Watts
        total_power_used += estimated_power_usage / 720  # Cumulative energy in Wh based on 5 sec intervals
        cpu_usage_sum += cpu_usage  # Accumulate CPU usage for averaging
        memory_usage_sum += memory_info.percent  # Accumulate memory usage for averaging

        # Compute the average CPU and memory usage so far
        avg_cpu_usage = cpu_usage_sum / interval_count
        avg_memory_usage = memory_usage_sum / interval_count

        # Calculate the average accuracy based on the cumulative confidence values
        avg_tc_accuracy = global_vars.total_tc_confidence / global_vars.prediction_count if global_vars.prediction_count > 0 else 0
        avg_vi_accuracy = global_vars.total_vi_confidence / global_vars.prediction_count if global_vars.prediction_count > 0 else 0

        # Update system stats table
        system_stats.append(html.Tr([html.Td("CPU Usage"), html.Td(f"{cpu_usage}%")]))
        system_stats.append(html.Tr([html.Td("Memory Usage"), html.Td(f"{memory_info.percent}%")]))
        system_stats.append(html.Tr([html.Td("Estimated Power Usage"), html.Td(f"{estimated_power_usage:.2f} Wh")]))
        system_stats.append(html.Tr([html.Td("Total Power Used (Wh)"), html.Td(f"{total_power_used:.2f} Watts")]))
        system_stats.append(html.Tr([html.Td("Average Tool Condition Accuracy"), html.Td(f"{avg_tc_accuracy:.2f}%")]))
        system_stats.append(html.Tr([html.Td("Average Visual Inspection Accuracy"), html.Td(f"{avg_vi_accuracy:.2f}%")]))
        system_stats.append(html.Tr([html.Td("Timestamp"), html.Td(time.strftime('%Y-%m-%d %H:%M:%S'))]))

        return figures + [terminal_output, feature_importance_figure, dbc.Table(system_stats, bordered=True, striped=True)]