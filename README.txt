
Author William Staal 
In Accordance with The University of Sheffield, 
where this repository acts as the main evidence for efforts done to achieve a Masters of Science in Electrical & Electronic Engineering.

# Tool Condition and Visual Inspection Monitoring Dashboard


## File Structure

├── simulate.py             # Simulates real-time data and controls the app's main execution flow(ACTING AS MAIN AS WELL)
├── app.py                  # Initializes the Dash app and defines the layout
├── callbacks.py            # Contains the callback functions for updating the dashboard
├── model_training.py       # Defines the functions for training the XGBoost models
├── global_vars.py          # Stores global variables shared across different modules
└──assets/                 # Contains any additional assets like CSS files

Note: Some variables will not accurately reflect the operation of the CNC machine. This can usually be detected by when M1_CURRENT_FEEDRATE reads 50, when X1 ActualPosition reads 198, or when M1_CURRENT_PROGRAM_NUMBER does not read 0. The source of these errors has not been identified.
Real_Time_Sim_1 consists of unworn -> worn & passes visual inspection -> worn & fails visual inspection
Real_Time_Sim_2 consists of worn & passes visual inspection(11) ->   worn & fails visual inspection (9)-> unworn(2)
Real_Time_Sim_3 consists of worn & fails visual inspection(6) -> consists of worn & passes visual inspection(14) -> worn & fails visual inspection(10)


Note: The XGBoost_ToolWear_Train.ipynb and XGBoost_VisualInspect_Train.ipynb act as preliminary broken down methods for training and optimising the XGBoost model for educational purposes.
Note: The Python_Testing_Platform_All_In_One.py embodies all functions in one singular running code if preferred.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher

### Installation

1.  Clone the Repository: 

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2.  Set Up a Virtual Environment: 

   If you're using Pipenv:

   ```bash
   pipenv install
   pipenv shell
   ```

   Otherwise, you can use `venv` and `pip`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3.  Install Dependencies: 

   ```bash
   pip install -r requirements.txt
   ```

4.  Download/Prepare Data: 

   Ensure you have your dataset ready. Place the CSV files (e.g., `train.csv`, `experiment_*.csv`, `Real_Time_Sim_*.csv`) in the appropriate directories or update the file paths in `simulate.py`.

### Running the Application

1.  Run the Application: 

   ```bash
   python simulate.py
   ```

   This will start the Dash app on `http://127.0.0.1:8050/`.

2.  Access the Dashboard: 

   Open a web browser and go to `http://127.0.0.1:8050/` to view the real-time monitoring dashboard.



## Overview

This repository contains a Python-based dashboard for monitoring tool conditions and visual inspections in real time using machine learning models. The dashboard is built using [Dash](https://dash.plotly.com/), [Plotly](https://plotly.com/), and [XGBoost](https://xgboost.readthedocs.io/). It visualizes data from real-time simulations, predicts tool wear and inspection outcomes, and displays system statistics such as CPU and memory usage.
BASED On CNC MILLING DATASET - UNIVERSITY OF MICHIGAN SMART LAB:  https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill/data?select=experiment_01.csv

A series of machining experiments were run on 2" x 2" x 1.5" wax blocks in a CNC milling machine in the System-level Manufacturing and Automation Research Testbed (SMART) at the University of Michigan. Machining data was collected from a CNC machine for variations of tool condition, feed rate, and clamping pressure. Each experiment produced a finished wax part with an "S" shape - S for smart manufacturing - carved into the top face, as shown in test_artifact.jpg

The dataset can be used in classification studies for:
(1) Tool wear detection
(2) Passing Visual Inspection
(3) Overall Tool Health based on the previous predictions

General data from a total of 18 different experiments are given in train.csv and includes:

Inputs (features)
No : experiment number
material : wax
feed_rate : relative velocity of the cutting tool along the workpiece (mm/s)
clamp_pressure : pressure used to hold the workpiece in the vise (bar)

Outputs (predictions)
tool_condition : label for unworn and worn tools
machining_completed : indicator for if machining was completed without the workpiece moving out of the pneumatic vise
passed_visual_inspection: indicator for if the workpiece passed visual inspection, only available for experiments where machining was completed



Time series data was collected from 18 experiments with a sampling rate of 100 ms and are separately reported in files experiment_01.csv to experiment_18.csv. Each file has measurements from the 4 motors in the CNC (X, Y, Z axes and spindle). These CNC measurements can be used in two ways:
(1) Taking every CNC measurement as an independent observation where the operation being performed is given in the Machining_Process column. Active machining operations are labeled as "Layer 1 Up", "Layer 1 Down", "Layer 2 Up", "Layer 2 Down", "Layer 3 Up", and "Layer 3 Down". 
(2) Taking each one of the 18 experiments (the entire time series) as an observation for time series classification

The features available in the machining datasets are:
X1_ActualPosition: actual x position of part (mm)
X1_ActualVelocity: actual x velocity of part (mm/s)
X1_ActualAcceleration: actual x acceleration of part (mm/s/s)
X1_CommandPosition: reference x position of part (mm)
X1_CommandVelocity: reference x velocity of part (mm/s)
X1_CommandAcceleration: reference x acceleration of part (mm/s/s)
X1_CurrentFeedback: current (A)
X1_DCBusVoltage: voltage (V)
X1_OutputCurrent: current (A)
X1_OutputVoltage: voltage (V)
X1_OutputPower: power (kW)

Y1_ActualPosition: actual y position of part (mm)
Y1_ActualVelocity: actual y velocity of part (mm/s)
Y1_ActualAcceleration: actual y acceleration of part (mm/s/s)
Y1_CommandPosition: reference y position of part (mm)
Y1_CommandVelocity: reference y velocity of part (mm/s)
Y1_CommandAcceleration: reference y acceleration of part (mm/s/s)
Y1_CurrentFeedback: current (A)
Y1_DCBusVoltage: voltage (V)
Y1_OutputCurrent: current (A)
Y1_OutputVoltage: voltage (V)
Y1_OutputPower: power (kW)

Z1_ActualPosition: actual z position of part (mm)
Z1_ActualVelocity: actual z velocity of part (mm/s)
Z1_ActualAcceleration: actual z acceleration of part (mm/s/s)
Z1_CommandPosition: reference z position of part (mm)
Z1_CommandVelocity: reference z velocity of part (mm/s)
Z1_CommandAcceleration: reference z acceleration of part (mm/s/s)
Z1_CurrentFeedback: current (A)
Z1_DCBusVoltage: voltage (V)
Z1_OutputCurrent: current (A)
Z1_OutputVoltage: voltage (V)

S1_ActualPosition: actual position of spindle (mm)
S1_ActualVelocity: actual velocity of spindle (mm/s)
S1_ActualAcceleration: actual acceleration of spindle (mm/s/s)
S1_CommandPosition: reference position of spindle (mm)
S1_CommandVelocity: reference velocity of spindle (mm/s)
S1_CommandAcceleration: reference acceleration of spindle (mm/s/s)
S1_CurrentFeedback: current (A)
S1_DCBusVoltage: voltage (V)
S1_OutputCurrent: current (A)
S1_OutputVoltage: voltage (V)
S1_OutputPower: current (A)
S1_SystemInertia: torque inertia (kg*m^2)

M1_CURRENT_PROGRAM_NUMBER: number the program is listed under on the CNC
M1_sequence_number: line of G-code being executed
M1_CURRENT_FEEDRATE: instantaneous feed rate of spindle

Machining_Process: the current machining stage being performed. Includes preparation, tracing up  and down the "S" curve involving different layers, and repositioning of the spindle as it moves through the air to a certain starting point

## Features

- Real-Time Monitoring: Simulate real-time data feeds to monitor tool conditions and visual inspections.
-  Machine Learning Integration:  Use trained XGBoost models to predict outcomes based on the data.
-  Visual Dashboards: Interactive dashboards displaying gauge charts, feature importance, system stats, and terminal outputs.
-  System Statistics:  Real-time CPU and memory usage, along with power consumption estimates.

