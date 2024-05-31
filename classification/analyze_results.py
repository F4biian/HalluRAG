import numpy as np
import os
import pandas as pd
import json

RESULTS_FILE_NAME = "baseline_results_long_prompt.json" # "baseline_results.json"
METRIC = ["test", "accuracy"] # auc_pr, auc_pr_hallucinated, auc_pr_grounded # ["y_test.mean"] # 
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

for model_name, quantization_dict in results.items():
    model_table = pd.DataFrame()
    for quantization_name, internal_states_dict in quantization_dict.items():
        for internal_state_name, clf_results in internal_states_dict.items():
            metric_values = np.array([get_deep_dict_value(METRIC, res) for res in clf_results])
            metric_mean = np.nanmean(metric_values)
            metric_std = np.nanstd(metric_values)

            model_table.at[internal_state_name, quantization_name] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)} ±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"

            if "13b" in model_name:
                if "_layer_100_last_" in internal_state_name:
                    if "float8" in quantization_name:
                        from pprint import pprint
                        t = []
                        for res in clf_results.copy():
                            res = res["test"]
                            for key in res.copy():
                                if type(res[key]) != float and type(res[key]) != int:
                                    if len(res[key]) > 5:
                                        del res[key]
                            t.append(res)
                        pprint(t)

    print()
    print(model_name)
    print(model_table)