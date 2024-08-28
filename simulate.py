# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:
# This script implements a real-time simulation for predicting the condition of machinery based on machine learning models. 
# It reads data from CSV files to simulate data sources, updates gauge values, and provides system statistics.
# The predictions are made using trained models for tool condition and visual inspection.
# The main function initializes the models and starts a Dash application for visualizing the results in real time.
import sys
import time
import random
from model_training import train_models
from utils import most_common
import pandas as pd
from system_stats import print_system_stats
import global_vars
###     READ CHUNKS FROM RealTime_sim1,2,3 TO SIMULATE DATA SOURCES AND UPDATE GAUGE VALUES  ###
import psutil
import time
from utils import Tee
import global_vars

def simulate_real_time_prediction(tc_model, vi_model, file_paths, labels, lines_per_second=100, output_interval=5):
    global total_tc_confidence, total_vi_confidence, prediction_count
    previous_condition_value = {label: None for label in labels}
    previous_inspection_value = {label: None for label in labels}

    # Initialize data iterators
    data_iters = {label: pd.read_csv(file_path, chunksize=lines_per_second * output_interval)
                  for label, file_path in zip(labels, file_paths)}

    while True:
        results = []

        for label in labels:
            chunk = next(data_iters[label], None)

            # If chunk is None, restart the iterator for the current label
            if chunk is None:
                data_iters[label] = pd.read_csv(file_paths[labels.index(label)], chunksize=lines_per_second * output_interval)
                chunk = next(data_iters[label])

            chunk = chunk.apply(pd.to_numeric, errors='coerce')
            chunk.fillna(0, inplace=True)
            features = chunk.drop(columns=['Machining_Process'], errors='ignore')

            # Predict using both models
            tc_y_pred = tc_model.predict(features)
            vi_y_pred = vi_model.predict(features)

            # Transform predictions into human-readable format
            condition_pred = ["Unworn" if pred == 0 else "Worn" for pred in tc_y_pred]
            inspection_pred = ["Passes Visual Inspection" if pred == 0 else "Fails Visual Inspection" for pred in vi_y_pred]

            # Calculate the most common prediction and confidence
            cond_val_t, cond_confidence = most_common(condition_pred)
            cond_val_v, inspec_confidence = most_common(inspection_pred)

            # Update global confidence accumulators and prediction count
            global_vars.total_tc_confidence += cond_confidence
            global_vars.total_vi_confidence += inspec_confidence
            global_vars.prediction_count += 1

            # Update condition value based on predictions
            if cond_val_t == "Unworn" and cond_val_v == "Passes Visual Inspection":
                condition_value = 17  # Green
                status_message = f"{label}: Tool is unworn and passes visual inspection. Confidence: {cond_confidence:.2f}% worn, {inspec_confidence:.2f}% visual inspection."
            elif cond_val_t == "Worn" and cond_val_v == "Passes Visual Inspection":
                condition_value = 50  # Amber
                status_message = f"{label}: Tool is worn but passes visual inspection. Confidence: {cond_confidence:.2f}% worn, {inspec_confidence:.2f}% visual inspection."
            elif cond_val_t == "Worn" and cond_val_v == "Fails Visual Inspection":
                condition_value = 80  # Red
                status_message = f"{label}: Tool is worn and fails visual inspection. Confidence: {cond_confidence:.2f}% worn, {inspec_confidence:.2f}% visual inspection."
            else:
                condition_value = previous_condition_value[label] if previous_condition_value[label] is not None else 50
                status_message = f"{label}: Error or mixed predictions occurred, using previous value."

            inspection_value = condition_value

            if condition_value != 0:
                previous_condition_value[label] = condition_value
            if inspection_value != 0:
                previous_inspection_value[label] = inspection_value

            results.append((condition_value, inspection_value, status_message, label))

            # Print system stats and top 10 features for both models
            print_system_stats(tc_model, vi_model, features, features)

        for result in results:
            yield result

def main(app):
    
    results_csv = 'train.csv'
    experiment_files = [f'experiment_{"0" + str(i) if i < 10 else str(i)}.csv' for i in range(1, 19)]
    file_paths = ["Real_Time_Sim_1.csv", "Real_Time_Sim_2.csv", "Real_Time_Sim_3.csv"]
    global_vars.labels = ["Machine 1", "Machine 2", "Machine 3"]

    print("Training models...")
    global_vars.tc_model, global_vars.vi_model, global_vars.tc_features, global_vars.vi_features = train_models(results_csv, experiment_files)

    print("Starting real-time simulation...")
    global_vars.generator =  simulate_real_time_prediction(global_vars.tc_model, global_vars.vi_model, file_paths, global_vars.labels)
    
    print("Running Dash app...")
    app.run_server(debug=False, port=8050)

if __name__ == "__main__":
    from app import app  # Import the Dash app from app.py
    with open("output_log.txt", "w") as log_file:
        tee = Tee(sys.stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee  # Also redirect stderr to catch errors
        try:
            main(app)
        finally:
            sys.stdout = sys.__stdout__  # Reset stdout to original
            sys.stderr = sys.__stderr__  # Reset stderr to original