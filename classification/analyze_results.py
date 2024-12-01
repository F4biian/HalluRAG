import numpy as np
import os
import pandas as pd
import json

RESULTS_FILE_NAME = "hallurag_results.json" # "hallurag_results_combinations_answerable_only.json" # "hallurag_results_combinations.json" # "test_ragtruth_on_hallurag_combinations.json" # "hallurag_results_combinations.json" # "baseline_results.json"
METRIC = ["test", "accuracy"] # ['train', 'val', 'test', 'test_random']  with  ['loss', 'cohen_kappa_threshold', 'cohen_kappa', 'mcc_threshold', 'mcc', 'accuracy_threshold', 'accuracy', 'confusion_matrix', 'f1_hallucinated_threshold', 'recall_hallucinated', 'precision_hallucinated', 'f1_hallucinated', 'fpr_hallucinated', 'tpr_hallucinated', 'roc_auc_hallucinated', 'P_hallucinated', 'R_hallucinated', 'auc_pr_hallucinated', 'f1_grounded_threshold', 'recall_grounded', 'precision_grounded', 'f1_grounded', 'fpr_grounded', 'tpr_grounded', 'roc_auc_grounded', 'P_grounded', 'R_grounded', 'auc_pr_grounded']
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
    model_table_shuffled = pd.DataFrame()
    for quantization_name, internal_states_dict in quantization_dict.items():
        for internal_state_name, clf_results in internal_states_dict["normal_target"].items():
            metric_values = np.array([get_deep_dict_value(METRIC, res) for res in clf_results])
            # print(model_name, quantization_name, internal_state_name, metric_values[0])

            if METRIC[-1] == "confusion_matrix":
                metric_values = metric_values.astype(np.float64)
                for i in range(metric_values.shape[0]):
                    metric_values[i, :] /= np.sum(metric_values[i, :])

                metric_mean = np.nanmean(metric_values, axis=0)
                metric_std = np.nanstd(metric_values, axis=0)

                metric_mean = np.round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)
                metric_std = np.round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)

                conf_str = f"{metric_mean[0][0]}±{metric_std[0][0]} {metric_mean[0][1]}±{metric_std[0][1]}\n{metric_mean[1][0]}±{metric_std[1][0]} {metric_mean[1][1]}±{metric_std[1][1]}"

                model_table.at[internal_state_name, quantization_name] = conf_str
            else:
                metric_mean = np.nanmean(metric_values)
                metric_std = np.nanstd(metric_values)

                model_table.at[internal_state_name, quantization_name] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"

        for internal_state_name, clf_results in internal_states_dict["shuffled_target"].items():
            metric_values = np.array([get_deep_dict_value(METRIC, res) for res in clf_results])
            metric_mean = np.nanmean(metric_values)
            metric_std = np.nanstd(metric_values)

            model_table_shuffled.at[internal_state_name, quantization_name] = f"{round(metric_mean * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}±{round(metric_std * METRIC_FACTOR, METRIC_ROUND_DECIMALS)}"

    print()
    if METRIC[-1] == "confusion_matrix":
        for col in model_table.columns:
            print(f"####### {col} #######")
            for index in model_table.index:
                print(f"------- {index} -------")
                print(model_table.at[index, col], "\n")
            print("\n\n")
    else:
        print(model_name, "NORMAL")
        print(model_table)
        print(model_name, "SHUFFLED")
        print(model_table_shuffled)