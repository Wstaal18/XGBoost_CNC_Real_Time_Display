# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:
# This script contains functions for creating visualizations using Plotly.
# It includes two main functions:
# 1. create_feature_importance_chart: Generates a bar chart displaying the top 10 features and their importance values.
# 2. create_gauge_chart: Creates a gauge chart representing the condition of a machine based on the condition and inspection values.


import plotly.graph_objects as go

def create_feature_importance_chart(overall_top_10_features):
    features, importances = zip(*overall_top_10_features)
    fig = go.Figure(go.Bar(
        x=features,
        y=importances,
        text=importances,
        textposition='auto',
        marker=dict(color='lightblue')
    ))
    fig.update_layout(title="Top 10 Feature Importances", xaxis_title="Feature", yaxis_title="Importance")
    return fig

def create_gauge_chart(condition_value, inspection_value, machine_label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=condition_value,
        title={'text': f"{machine_label} Condition"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "green"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ]
        }
    ))
    return fig
