# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: [23/08/2024]

# Description:
# This script defines a function `print_system_stats` that retrieves and displays system statistics including:
    # CPU usage, memory usage, and estimated power usage. 
# It also calculates and prints the top 10 features from two machine learning models:
    # a tool condition model (tc_model) and a visual inspection model (vi_model).
# The function utilizes the psutil library to gather system information and prints
# the results to the console in a formatted manner.



import psutil
import time
from utils import get_overall_top_10_features

def print_system_stats(tc_model, vi_model, tc_features, vi_features):
    CPU_TDP = 45  # Adjust this value to your CPU's TDP

    cpu_usage = psutil.cpu_percent(interval=0)
    memory_info = psutil.virtual_memory()
    estimated_power_usage = (cpu_usage / 100) * CPU_TDP

    tc_features_importances = [(tc_features.columns[i], v) for i, v in enumerate(tc_model.feature_importances_)]
    vi_features_importances = [(vi_features.columns[i], v) for i, v in enumerate(vi_model.feature_importances_)]

    overall_top_10_features = get_overall_top_10_features(tc_features_importances, vi_features_importances)
    print(f"\n--- Overall Top 10 Features ---")
    for feature, importance in overall_top_10_features:
        print(f"{feature}: {importance:.4f}")

    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_info.percent}%")
    print(f"Estimated Power Usage: {estimated_power_usage:.2f} Watts")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
