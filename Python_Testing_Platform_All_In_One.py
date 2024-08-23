from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import psutil
import time
from collections import Counter

######     CREATION & PRINT FUNCTIONS ######
###     CREATE DASH APPLICATION LAYOUT ###
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Tool Condition and Visual Inspection Monitoring", className='mb-2', style={'textAlign': 'center'}),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='gauge-chart-1', config={"displayModeBar": False}),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='feature-importance-chart', config={"displayModeBar": False})
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='gauge-chart-2', config={"displayModeBar": False}),
        ], width=6),

        dbc.Col([
            html.H3("System Stats"),
            dbc.Table(id="system-stats-table", bordered=True, striped=True)
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='gauge-chart-3', config={"displayModeBar": False}),
        ], width=6),

         dbc.Col([
            html.H3("Terminal Output"),
            html.Pre(id="terminal-output", style={"background-color": "#f8f9fa", "padding": "10px", "border": "1px solid #ccc"})
        ], width=6)
    ]),
    
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Update every 5 seconds
])

###     TRAIN XGBOOST MODELS ###
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

###     CREATE FEATURE IMPORTANCE CHART ###

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

###     CREATE FEATURE GAUGE CHART'S ###

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

###     CALCULATE SYSTEM STATISTICS & PRINT TO TERMINAL & DASH APPLICATION ###
def print_system_stats(tc_model, vi_model, tc_features, vi_features):
    CPU_TDP = 45  # Adjust this value to your CPU's TDP (e.g., 13th Gen Intel Core i7-13620H)

    # Set interval to 0 to avoid unintended delays
    cpu_usage = psutil.cpu_percent(interval=0)
    memory_info = psutil.virtual_memory()

    # Calculate the estimated power usage
    estimated_power_usage = (cpu_usage / 100) * CPU_TDP

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
    print(f"Estimated Power Usage: {estimated_power_usage:.2f} Watts")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")




######     SUPLEMENTARY FUNCTIONS ######

###     CALCULATE MOST COMMON OUTPUT FUNCTION FOR FEATURE IMPORTANCE ###
from collections import Counter

def most_common(lst):
    """Return the most common element in a list and its confidence percentage."""
    data = Counter(lst)
    most_common_element, count = data.most_common(1)[0]
    confidence = count / len(lst) * 100  # Calculate confidence as a percentage
    return most_common_element, confidence


###     FIND MOST MOST IMPORTANT FEATURES FUNCTION ###
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


######     UPDATING FUNCTIONS ######

###     READ CHUNKS FROM RealTime_sim1,2,3 TO SIMULATE DATA SOURCES AND UPDATE GAUGE VALUES  ###
def simulate_real_time_prediction(tc_model, vi_model, file_paths, labels, lines_per_second=100, output_interval=5):
    previous_condition_value = {label: None for label in labels}
    previous_inspection_value = {label: None for label in labels}
    data_iters = {label: pd.read_csv(file_path, chunksize=lines_per_second * output_interval)
                  for label, file_path in zip(labels, file_paths)}

    while True:
        results = []

        for label in labels:
            chunk = next(data_iters[label], None)

            if chunk is not None:
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



###     APP CALLBACK FUNCTION  ###
@app.callback(
    [Output(f'gauge-chart-{i}', 'figure') for i in range(1, 4)] +
    [Output("terminal-output", "children"),
     Output('feature-importance-chart', 'figure'),
     Output('system-stats-table', 'children')],
    Input('interval-component', 'n_intervals')
)

###     UPDATE/REDRAW GAUGE CHART BASED ON UPDATED VALUES###
def update_gauge_charts(n_intervals):
    global tc_features, vi_features
    figures = []
    terminal_output = ""
    system_stats = []

    # Generate the gauge charts
    for condition_value, inspection_value, status_message, label in generator:
        index = labels.index(label)
        figures.append(create_gauge_chart(condition_value, inspection_value, label))
        terminal_output += status_message + "\n"
        if len(figures) == len(labels):
            break

    # Generate the feature importance chart
    overall_top_10_features = get_overall_top_10_features(
        [(tc_features.columns[i], v) for i, v in enumerate(tc_model.feature_importances_)],
        [(vi_features.columns[i], v) for i, v in enumerate(vi_model.feature_importances_)]
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

    return figures + [terminal_output, feature_importance_figure, dbc.Table(system_stats, bordered=True, striped=True)]


#####   MAIN FUNCTION TO CONTROL FULL CODE  #####
def main():
    global generator, labels, tc_model, vi_model, tc_features, vi_features
    
    results_csv = 'train.csv'
    experiment_files = [f'experiment_{"0" + str(i) if i < 10 else str(i)}.csv' for i in range(1, 19)]
    file_paths = ["Real_Time_Sim_1.csv", "Real_Time_Sim_2.csv", "Real_Time_Sim_3.csv"]
    labels = ["Machine 1", "Machine 2", "Machine 3"]

    print("Training models...")
    tc_model, vi_model, tc_features, vi_features = train_models(results_csv, experiment_files)

    print("Starting real-time simulation...")
    generator = simulate_real_time_prediction(tc_model, vi_model, file_paths, labels)
    
    print("Running Dash app...")
    app.run_server(debug=False, port=8050)

if __name__ == "__main__":
    main()
