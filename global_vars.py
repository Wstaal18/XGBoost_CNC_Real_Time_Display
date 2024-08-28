# Initialize global variables to None
tc_model = None
vi_model = None
tc_features = None
vi_features = None
labels = None
generator = None
# Add global variables for tracking total power and usage counts
total_power_used = 0.0
cpu_usage_sum = 0.0
memory_usage_sum = 0.0
interval_count = 0
# Add global variables for tracking total confidence and count of predictions
total_tc_confidence = 0.0
total_vi_confidence = 0.0
prediction_count = 0
CPU_TDP = 45  # Adjust this value to your CPU's TDP
