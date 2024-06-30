import numpy as np
import os
import pandas as pd
import json

RESULTS_FILE_NAME = "hallurag_robustness_results.json" # "baseline_results.json"
METRIC = ["train", "accuracy"]
METRIC1 = ["val", "accuracy"]
METRIC2 = ["test", "accuracy"]
METRIC_FACTOR = 100
METRIC_ROUND_DECIMALS = 2

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_FILE = os.path.join(CURR_DIR, RESULTS_FILE_NAME)

# Load results from json file
# structure of results:  :LLM name: -> :quantization name: -> :internal state name: -> :train/val/test results of 15 classifiers:
with open(RESULTS_FILE, "r") as file:
    results = json.load(file)

def get_deep_dict_value(path_to_go, current):
    if len(path_to_go) == 0:
        return current
    else:
        return get_deep_dict_value(path_to_go[1:], current[path_to_go[0]])

for param_name, param_dict in results.items():
    param_table = pd.DataFrame()
    for param_value, param_results in param_dict.items():
            metric_values = np.array([get_deep_dict_value(METRIC, res) for res in param_results])
            metric_mean = np.nanmean(metric_values)
            metric_std = np.nanstd(metric_values)
            param_table.at[param_value, "train"] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)} ±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"
            
            metric_values = np.array([get_deep_dict_value(METRIC1, res) for res in param_results])
            metric_mean = np.nanmean(metric_values)
            metric_std = np.nanstd(metric_values)
            param_table.at[param_value, "val"] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)} ±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"
            
            metric_values = np.array([get_deep_dict_value(METRIC2, res) for res in param_results])
            metric_mean = np.nanmean(metric_values)
            metric_std = np.nanstd(metric_values)
            param_table.at[param_value, "test"] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)} ±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"
            
    print()
    print("###", param_name)
    print(param_table)
    print()
    print()