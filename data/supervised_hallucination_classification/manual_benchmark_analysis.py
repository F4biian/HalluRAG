########################################################################################
# IMPORTS

import os
import pandas as pd
import json
from pprint import pprint
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
FILE = os.path.join(CURR_DIR, "manual_benchmark_results4o.json")

with open(FILE, "r") as file:
    results = json.load(file)

def get_target_from_manual(answerable, sent_dict):
    str_target = sent_dict["target"]

    if answerable:
        if str_target == "W" or str_target == "IDK":
            target = 1
        elif str_target == "R":
            target = 0
        else:
            return None
    else:
        if str_target == "W":
            target = 1
        elif str_target == "IDK":
            target = 0
        else:
            return None
    
    return target

def get_shd_prediction(answerable, pred):
    if pred is None:
        return None

    conflicting_fail_content = pred["conflicting_fail_content"]
    conflicting_fail = pred["conflicting_fail"]
    grounded_fail_content = pred["grounded_fail_content"]
    grounded_fail = pred["grounded_fail"]
    no_clear_answer_fail_content = pred["no_clear_answer_fail_content"]
    no_clear_answer_fail = pred["no_clear_answer_fail"]

    has_fail = conflicting_fail_content or conflicting_fail or grounded_fail_content or grounded_fail or no_clear_answer_fail_content or no_clear_answer_fail
    if has_fail:
        return None

    conflicting = pred["conflicting"]
    grounded = pred["grounded"]
    has_factual_information = pred["has_factual_information"]
    no_clear_answer = pred["no_clear_answer"]

    if conflicting is None or grounded is None or has_factual_information is None or no_clear_answer is None:
        return None

    if grounded and has_factual_information and no_clear_answer:
        return None

    if answerable:
        if conflicting:
            prediction = 1
        elif has_factual_information and grounded:
            prediction = 0
        elif no_clear_answer:
            prediction = 1
        else:
            if has_factual_information:
                if grounded:
                    prediction = 0
                else:
                    prediction = 1
            else:
                prediction = 0
    else:
        if conflicting:
            prediction = 1
        elif not grounded and has_factual_information:
            prediction = 1
        elif no_clear_answer:
            prediction = 0
        else:
            if has_factual_information:
                if grounded:
                    print("-"*10)
                    pprint(pred["llm_eval"])
                    print("-"*10)
                    return None
                    # raise Exception("Unanswerable question has been answered. Should not be possible!")
                else:
                    prediction = 1
            else:
                prediction = 0
    return prediction

data = []
for res in results:
    answerable = res["prompt"]["answerable"]
    for sent_i, sent_dict in enumerate(res["sentence_data"]):
        pred = sent_dict["pred"]

        target = get_target_from_manual(answerable, sent_dict)
        shd_pred = get_shd_prediction(answerable, pred)

        if target is None or shd_pred is None:
            print("None")
            continue

        # if target != prediction:
        #     pprint(pred["llm_eval"])
        #     print("target:", target)
        #     print("prediction:", prediction)
        #     print("answerable:", answerable)
        #     print("human:", [s["target"] for s in res["sentence_data"]][sent_i])
        #     input()

        data.append({
            "answerable": answerable,
            "model": res["model"].split("/", 1)[-1],
            "model_quant": res["model"].split("/", 1)[-1] + (f" ({res['quantization']})" if res["quantization"] else ""),
            "prediction": shd_pred,
            "target": target
        })

df = pd.DataFrame(data)

for model_name, model_df in df.groupby("model_quant"):
    print("="*50)
    print("Model name:", model_name)
    print("Length:", model_df.shape[0])
    print("Target mean:", model_df["target"].mean())
    print("Prediction mean:", model_df["prediction"].mean())

    print("Acc:", (model_df["prediction"] == model_df["target"]).mean())
    print("Corr:", model_df["prediction"].corr(model_df["target"]))

    print("κ:", cohen_kappa_score(model_df["target"], model_df["prediction"]))
    print("Conf matrix:\n", confusion_matrix(model_df["target"], model_df["prediction"]))
    # [[ True Negative   False positive] [ False Negative   True Positive ]]
    # [[ True Grounded   False Hallu] [ False Grounded   True Hallu ]]

model_df = df
model_name = "All"

print("="*50)
print("Model name:", model_name)
print("Length:", model_df.shape[0])
print("Target mean:", model_df["target"].mean())
print("Prediction mean:", model_df["prediction"].mean())

print("Acc:", (model_df["prediction"] == model_df["target"]).mean())
print("Corr:", model_df["prediction"].corr(model_df["target"]))

print("κ:", cohen_kappa_score(model_df["target"], model_df["prediction"]))
print("p:", precision_score(model_df["target"], model_df["prediction"]))
print("f1:", f1_score(model_df["target"], model_df["prediction"]))
print("Conf matrix:\n", confusion_matrix(model_df["target"], model_df["prediction"]))
# [[ True Negative   False positive]
# [ False Negative   True Positive ]]

# [[ True Grounded   False Hallu]
# [ False Grounded   True Hallu ]]