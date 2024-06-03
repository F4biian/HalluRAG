########################################################################################
# IMPORTS

import os
import pandas as pd
import json

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
FILE = os.path.join(CURR_DIR, "benchmark_results_3.5.json")

with open(FILE, "r") as file:
    results = json.load(file)

df = pd.DataFrame(results)

for model_name, model_df in df.groupby("model"):
    print("="*50)
    print("Model name:", model_name)
    print("Target mean:", model_df["target"].mean())

    unknowns = model_df["prediction"] == "UNKNOWN"
    print("Unknowns:", unknowns.sum())

    model_df = model_df[~unknowns]

    preds1 = model_df["prediction"] != "CORRECT"
    preds2 = model_df["prediction"] == "INCORRECT"

    print("Prediction   (!= CORRECT) mean:", preds1.mean())
    print("Prediction (== INCORRECT) mean:", preds2.mean())

    print("Prediction   (!= CORRECT) acc:", (preds1 == (model_df["target"] == 1)).mean())
    print("Prediction (== INCORRECT) acc:", (preds2 == (model_df["target"] == 1)).mean())

    print("Prediction   (!= CORRECT) corr:", (preds1.corr(model_df["target"] == 1)))
    print("Prediction (== INCORRECT) corr:", (preds2.corr(model_df["target"] == 1)))


model_df = df
model_name = "All"

print("="*50)
print("Model name:", model_name)
print("Target mean:", model_df["target"].mean())

unknowns = model_df["prediction"] == "UNKNOWN"
print("Unknowns:", unknowns.sum())

model_df = model_df[~unknowns]

preds1 = model_df["prediction"] != "CORRECT"
preds2 = model_df["prediction"] == "INCORRECT"

print("Prediction   (!= CORRECT) mean:", preds1.mean())
print("Prediction (== INCORRECT) mean:", preds2.mean())

print("Prediction   (!= CORRECT) acc:", (preds1 == (model_df["target"] == 1)).mean())
print("Prediction (== INCORRECT) acc:", (preds2 == (model_df["target"] == 1)).mean())

print("Prediction   (!= CORRECT) corr:", (preds1.corr(model_df["target"] == 1)))
print("Prediction (== INCORRECT) corr:", (preds2.corr(model_df["target"] == 1)))

