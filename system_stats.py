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
import global_vars

total_power_used = 0.0
cpu_usage_sum = 0.0
memory_usage_sum = 0.0
interval_count = 0
# Add global variables for tracking total confidence and count of predictions
total_tc_confidence = 0.0
total_vi_confidence = 0.0
prediction_count = 0
CPU_TDP = 45  # Adjust this value to your CPU's TDP

def print_system_stats(tc_model, vi_model, tc_features, vi_features):
    global total_power_used, cpu_usage_sum, memory_usage_sum, interval_count
    global estimated_power_usage
    
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

    # Get and sort feature importances for tc_model (Tool Condition Model)
    tc_features_importances = [(tc_features.columns[i], v) for i, v in enumerate(tc_model.feature_importances_)]
    tc_features_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Get and sort feature importances for vi_model (Visual Inspection Model)
    vi_features_importances = [(vi_features.columns[i], v) for i, v in enumerate(vi_model.feature_importances_)]
    vi_features_importances.sort(key=lambda x: x[1], reverse=True)

    # Get overall top 10 features
    overall_top_10_features = get_overall_top_10_features(tc_features_importances, vi_features_importances)
    print(f"\n--- Overall Top 10 Features ---")
    for feature, importance in overall_top_10_features:
        print(f"{feature}: {importance:.4f}")

    # Print system stats
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_info.percent}%")
    print(f"Estimated Power Usage: {estimated_power_usage:.2f} Wh")
    print(f"Total Power Used So Far: {total_power_used:.2f} Watts")  # Changed to Wh
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average Memory Usage: {avg_memory_usage:.2f}%")
    print(f"Average Tool Condition Accuracy: {avg_tc_accuracy:.2f}%")
    print(f"Average Visual Inspection Accuracy: {avg_vi_accuracy:.2f}%")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

