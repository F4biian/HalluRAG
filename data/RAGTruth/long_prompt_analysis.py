import numpy as np
import os
import pickle
import pandas as pd
import json

from pprint import pprint

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
LONG_PROMPT_INTERNAL_STATES_DIR = os.path.join(CURR_DIR, "long_prompt_internal_states")
INTERNAL_STATES_DIR = os.path.join(CURR_DIR, "internal_states")

LONG_FILE = os.path.join(LONG_PROMPT_INTERNAL_STATES_DIR, "mistralai_Mistral-7B-Instruct-v0.1 (float8).pickle")
SHORT_FILE = os.path.join(INTERNAL_STATES_DIR, "mistralai_Mistral-7B-Instruct-v0.1 (float8).pickle")

short_prompts = []
short_responses = []
long_prompts = []
long_responses = []

with open(LONG_FILE, 'rb') as handle:
    file_json = pickle.load(handle)
    for passage_data in file_json:
        long_prompts.append(len(passage_data["prompt"]))
        for sentence_data in passage_data["sentence_data"]:
            long_responses.append(len(sentence_data["cum_sentence"]))

with open(SHORT_FILE, 'rb') as handle:
    file_json = pickle.load(handle)
    for passage_data in file_json:
        short_prompts.append(len(passage_data["prompt"]) + len(passage_data["sentence_data"][-1]["cum_sentence"]))
        for sentence_data in passage_data["sentence_data"]:
            short_responses.append(len(sentence_data["cum_sentence"]))

short_prompts = pd.Series(short_prompts)
short_responses = pd.Series(short_responses)
long_prompts = pd.Series(long_prompts)
long_responses = pd.Series(long_responses)

print(short_prompts.describe())
"""
count     750.000000
mean     2732.558667
std       884.853376
min       979.000000
25%      1998.000000
50%      2717.500000
75%      3273.000000
max      5423.000000
dtype: float64
"""

print(short_responses.describe())
"""
count    4719.000000
mean      404.676838
std       245.371005
min         2.000000
25%       207.500000
50%       366.000000
75%       555.500000
max      1624.000000
dtype: float64
"""
print(long_prompts.describe())
"""
count      700.000000
mean      4307.854286
std       1728.552654
min       1119.000000
25%       3059.000000
50%       3832.000000
75%       5171.250000
max      10719.000000
dtype: float64
"""

print(long_responses.describe())
"""
count    6067.000000
mean      548.434317
std       355.598662
min        11.000000
25%       266.000000
50%       487.000000
75%       765.000000
max      2102.000000
dtype: float64
"""


best_thr = None
best_dist = 0

for i in range(max(long_prompts.max(), short_prompts.max())):
    remaining_long = len(long_prompts[long_prompts >= i])
    remaining_short = len(short_prompts[short_prompts < i])

    dist = abs(remaining_long - remaining_short)
    if best_thr is None or best_dist > dist:
        best_dist = dist
        best_thr = i

print(best_thr) # 3112
print(best_dist) # 0
print(long_prompts[long_prompts >= best_thr].describe())
"""
count      512.000000
mean      4925.550781
std       1613.967126
min       3112.000000
25%       3702.000000
50%       4500.500000
75%       5672.000000
max      10719.000000
dtype: float64
"""

print(short_prompts[short_prompts < best_thr].describe())
"""
count     512.000000
mean     2261.152344
std       550.447884
min       979.000000
25%      1771.750000
50%      2325.000000
75%      2744.750000
max      3109.000000
dtype: float64
"""