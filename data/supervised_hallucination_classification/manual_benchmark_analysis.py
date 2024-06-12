########################################################################################
# IMPORTS

import os
import pandas as pd
import json
from pprint import pprint

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
FILE = os.path.join(CURR_DIR, "manual_benchmark_results.json")

with open(FILE, "r") as file:
    results = json.load(file)

data = []
for res in results:
    answerable = res["prompt"]["answerable"]
    for sent_i, sent_dict in enumerate(res["sentence_data"]):
        pred = sent_dict["pred"]

        conflicting_fail_content = pred["conflicting_fail_content"]
        conflicting_fail = pred["conflicting_fail"]
        grounded_fail_content = pred["grounded_fail_content"]
        grounded_fail = pred["grounded_fail"]
        no_clear_answer_fail_content = pred["no_clear_answer_fail_content"]
        no_clear_answer_fail = pred["no_clear_answer_fail"]

        has_fail = conflicting_fail_content or conflicting_fail or grounded_fail_content or grounded_fail or no_clear_answer_fail_content or no_clear_answer_fail
        if has_fail:
            continue

        conflicting = pred["conflicting"]
        grounded = pred["grounded"]
        has_factual_information = pred["has_factual_information"]
        no_clear_answer = pred["no_clear_answer"]

        if conflicting is None or grounded is None or has_factual_information is None or no_clear_answer is None:
            continue

        str_target = sent_dict["target"]

        if answerable:
            if str_target == "W" or str_target == "IDK":
                target = 1
            elif str_target == "R":
                target = 0
            else:
                continue
        else:
            if str_target == "W":
                target = 1
            elif str_target == "IDK":
                target = 0
            else:
                continue

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
            elif no_clear_answer:
                prediction = 0
            else:
                if has_factual_information:
                    if grounded:
                        print("-"*10)
                        pprint(pred["llm_eval"])
                        print("-"*10)
                        continue
                        # raise Exception("Unanswerable question has been answered. Should not be possible!")
                    else:
                        prediction = 1
                else:
                    prediction = 0

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
            "prediction": prediction,
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


model_df = df
model_name = "All"

print("="*50)
print("Model name:", model_name)
print("Length:", model_df.shape[0])
print("Target mean:", model_df["target"].mean())
print("Prediction mean:", model_df["prediction"].mean())

print("Acc:", (model_df["prediction"] == model_df["target"]).mean())
print("Corr:", model_df["prediction"].corr(model_df["target"]))