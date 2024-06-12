########################################################################################
# IMPORTS

import fasttext
import os
import json
import re
import pandas as pd

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
TRAIN_FILE = os.path.join(CURR_DIR, "idk.train")
TEST_FILE = os.path.join(CURR_DIR, "idk.test")
MODEL_FILE = os.path.join(CURR_DIR, "idk_model.bin")
BENCHMARK_DATA_FILE = os.path.join(CURR_DIR, "..", "manual_classification.json")
TRAIN_PERC = 0.75
CLEAN_REGEX = r'\d|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~'

def create_files(train_perc=0.75) -> None:
    # TODO: implement once data exist

    data = []
    with open(BENCHMARK_DATA_FILE, "r") as file:
        data = json.load(file)["data"]

    all_sent_count = 0
    for d in data:
        all_sent_count += len(d["sentence_data"])

    train_data = []

    for d in data:
        answerable = d["prompt"]["answerable"]

        # print(d["model"])
        # print(d["quantization"])

        last_sent_end = 0
        for sent_i, sent_data in enumerate(d["sentence_data"]):
            sent = sent_data['cum_sentence'][last_sent_end:].strip()
            human_label = sent_data["target"]
            
            if answerable:
                if human_label == "W" or human_label == "IDK":
                    target = 1
                elif human_label == "R":
                    target = 0
                else:
                    continue
            else:
                if human_label == "W" or human_label == "IDK":
                    target = 1
                elif human_label == "R":
                    target = 0
                else:
                    continue

            ft_sent = re.sub(CLEAN_REGEX, '', sent.lower())
            ft_sent = re.sub(r'\s', ' ', ft_sent.strip())

            ft_data = {
                "target": target,
                "label": human_label,
                "ft_label": f"__label__{'idk' if human_label == 'IDK' else 'ik'}",
                "sent": sent,
                "ft_sent": ft_sent
            }
            train_data.append(ft_data)

            last_sent_end = len(sent_data['cum_sentence'])
        
    train_df = pd.DataFrame(train_data)
    test_df = train_df.iloc[int(train_df.shape[0]*train_perc):]
    train_df = train_df.iloc[:int(train_df.shape[0]*train_perc)]

    print(train_df)
    print(train_df["ft_label"].value_counts())
    print(test_df)
    print(test_df["ft_label"].value_counts())

    train_file_content = ""
    for _, row in train_df.iterrows():
        train_file_content += f"{row['ft_label']} {row['ft_sent']}\n"
    train_file_content = train_file_content.strip()
    with open(TRAIN_FILE, "w") as file:
        file.write(train_file_content)

    test_file_content = ""
    for _, row in test_df.iterrows():
        test_file_content += f"{row['ft_label']} {row['ft_sent']}\n"
    test_file_content = test_file_content.strip()
    with open(TEST_FILE, "w") as file:
        file.write(test_file_content)

create_files()

# Train model
model = fasttext.train_supervised(input=TRAIN_FILE, lr=0.3, epoch=50, wordNgrams=2, dim=50) # , lr=1.0, epoch=25, wordNgrams=2, dim=50

# Show train results
train_results = model.test(TRAIN_FILE)
print(f"Train Results: {train_results}")

# Test model
test_results = model.test(TEST_FILE)
print(f"Test Results: {test_results}")

# Save model
if input(f"Save model to {MODEL_FILE}? [y|anything else for no]").lower().strip() == "y":
    model.save_model(MODEL_FILE)
    print("Saved!")