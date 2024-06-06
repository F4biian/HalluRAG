########################################################################################
# IMPORTS

import os
import warnings
import pandas as pd
import json
from typing import List
from tqdm import tqdm
from models.utils import old_sentence_split
from pprint import pprint
import pickle
import random

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
INTERNAL_STATES_DIR = os.path.join(CURR_DIR, "..", "qna2output", "internal_states")
QNA2OUTPUT_DIR = os.path.join(CURR_DIR, "..", "qna2output")
RESULTS_FILE = os.path.join(CURR_DIR, "manual_classification.json")
MAX_ROUNDS = 100

random.seed(432)

responses = {}

for file in os.listdir(INTERNAL_STATES_DIR):
    model_name = file.split("_", 1)[-1].replace(".pickle", "")
    if file.startswith("train"):
        with open(os.path.join(INTERNAL_STATES_DIR, file), 'rb') as handle:
            file_json = pickle.load(handle)
            random.shuffle(file_json)
            responses[model_name] = file_json

data = []
feedback_given_counter = 0
counter_per_llm = {model_name: -1 for model_name in responses}
current_round = 0

def load():
    global data, feedback_given_counter, counter_per_llm, current_round
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as file:
            d = json.load(file)
            data = d["data"]
            feedback_given_counter = d["feedback_given_counter"]
            counter_per_llm = d["counter_per_llm"]
            current_round = d["current_round"]

def save(use_indent=False):
    d = {
        "feedback_given_counter": feedback_given_counter,
        "counter_per_llm": counter_per_llm,
        "data": data,
        "current_round": current_round
    }
    with open(RESULTS_FILE, "w") as file:
        if use_indent:
            json.dump(d, file, indent=4, default=str)
        else:
            json.dump(d, file, default=str)

load()

while current_round < MAX_ROUNDS:
    for model_name, model_data in responses.items():
        counter_per_llm[model_name] += 1

        source_data = model_data[counter_per_llm[model_name]]
        answerable = source_data['prompt']['answerable']

        print("\n"*10)
        print(f"[{current_round+1}]", "="*45)
        print(f"     Model: {model_name}")
        print(f"    qna_id: {source_data['prompt']['passage']['useful_art_i']}_{source_data['prompt']['passage']['useful_passage_i']}")
        print(f"    titles: {' ### '.join([t['article_title'] for t in source_data['prompt']['other_passages']])}")
        print(f"     title: {source_data['prompt']['passage']['article_title']}")
        print(f"   #chunks: {source_data['prompt']['chunks_per_prompt']}")
        print(f"answerable: {answerable}")
        print("- "*25)
        print()

        print(f"    QUESTION: {source_data['prompt']['passage']['question']}")
        print(f"ANSWER QUOTE: {source_data['prompt']['passage']['answer_quote']}")
        print()
        # print(f"CONTEXT:\n{source_data['prompt']['other_passages'][source_data['prompt']['answer_chunk_index']]}")
        print(f"CONTEXT:\n{source_data['prompt']['passage']['context']}")
        print()
        print(f"PASSAGE:\n{source_data['prompt']['passage']['passage_text']}")
        print("- "*25)

        last_sent_end = 0
        for sent_i, sent_data in enumerate(source_data["sentence_data"]):
            print(f"SENTENCE: {sent_data['cum_sentence'][last_sent_end:].strip()}")
            last_sent_end = len(sent_data['cum_sentence'])

            user = input(f"[{feedback_given_counter}] idk|right|wrong: ").strip().lower()
            if user.startswith("i") or user == "1":
                label = "IDK"
            elif user.startswith("r") or user == "2":
                label = "R"
            elif user.startswith("w") or user == "3":
                label = "W"
            elif user.startswith("x"):
                save(True)
                exit()
            else:
                continue

            print("->", label)
            source_data["sentence_data"][sent_i]["target"] = label
            feedback_given_counter += 1
            print()
        
        data.append(source_data)
        save()
    current_round += 1
    save()