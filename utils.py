# Author: William Staal
# University: University of Sheffield
# Course: MSc in Electrical and Electronic Engineering
# Project: Dissertation Project
# Date: 23/08/2024

# Description:

# This script defines two functions: `most_common` and `get_overall_top_10_features`.
#The `most_common` function calculates and returns the most common element in a listalong with its confidence percentage.
#The `get_overall_top_10_features` function combines feature importances from two machine learning models (tool condition and
#visual inspection models) and returns the top 10 most important features based on their combined importance scores.

from collections import Counter

def most_common(lst):
    #Return the most common element in a list and its confidence percentage."""
    data = Counter(lst)
    most_common_element, count = data.most_common(1)[0]
    confidence = count / len(lst) * 100  # Calculate confidence as a percentage
    return most_common_element, confidence

def get_overall_top_10_features(tc_features_importances, vi_features_importances):
    # Combine the feature importances from both models
    combined_importances = {}

    # Add tc_model features
    for feature, importance in tc_features_importances:
        if feature in combined_importances:
            combined_importances[feature] += importance
        else:
            combined_importances[feature] = importance

    # Add vi_model features
    for feature, importance in vi_features_importances:
        if feature in combined_importances:
            combined_importances[feature] += importance
        else:
            combined_importances[feature] = importance

    # Sort the combined features by importance in descending order
    sorted_combined_importances = sorted(combined_importances.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 features
    top_10_features = sorted_combined_importances[:10]

    return top_10_features