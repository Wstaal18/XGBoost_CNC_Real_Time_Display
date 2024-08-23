# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:
# This script contains a function to train machine learning models using the XGBoost classifier. 
# It reads data from CSV files to train two models:
    # 1. Tool Condition Model: Predicts whether a tool is worn based on features.
    # 2. Visual Inspection Model: Predicts whether an item passes visual inspection.
# The function train_models consolidates data from multiple experiment files,
# prepares the data for training, and returns the trained models along with their respective feature sets for further analysis.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_models(results_csv, experiment_files):
    def train_model(target_col, target_value):
        frames = []
        results = pd.read_csv(results_csv)
        for i, file in enumerate(experiment_files):
            frame = pd.read_csv(file)
            row = results[results['No'] == i + 1]
            frame['target'] = 1 if row.iloc[0][target_col] == target_value else 0
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.drop(columns=['Machining_Process'], inplace=True)
        df.fillna(0, inplace=True)
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=100)
        model = xgb.XGBClassifier(seed=42, objective='binary:logistic', gamma=0.025, learning_rate=0.1,
                                  max_depth=5, reg_lambda=5, scale_pos_weight=2, subsample=0.83, colsample_bytree=0.7)
        model.fit(X_train, y_train, verbose=True, eval_set=[(X_test, y_test)])
        
        # Return the model and the training features for feature importance
        return model, X_train

    tc_model, tc_features = train_model('tool_condition', 'worn')
    vi_model, vi_features = train_model('passed_visual_inspection', 'no')
    return tc_model, vi_model, tc_features, vi_features