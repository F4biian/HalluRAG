########################################################################################
# IMPORTS

import os
import warnings
import pandas as pd
import json
from typing import List
from tqdm import tqdm
from models.utils import sentence_split, cum_concat
from SHD.shd import classify
from pprint import pprint
import re

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
RAGTRUTH_DIR = os.path.join(CURR_DIR, "..", "RAGTruth")
SAMPLES_PER_LLM = 2 # 3 llms
SIMPLIFICATION_REGEX = r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~'

def get_targets(sentence_start_indices, labels) -> List[int]:
    # Assign each sentence a 0 (non-hallucinated) or 1 (hallucinated).
    sentence_targets = [0 for _ in sentence_start_indices]
    for hallucination_label in labels:
        # Finding the sentence where the hallucination starts
        hallu_starts_in_sentence = 0
        for i in range(len(sentence_start_indices)):
            if hallucination_label["start"] < sentence_start_indices[i]:
                hallu_starts_in_sentence = i-1
                break
        else:
            hallu_starts_in_sentence = len(sentence_start_indices)-1

        # Finding the sentence where the hallucination ends
        hallu_ends_in_sentence = 0
        for i in range(hallu_starts_in_sentence, len(sentence_start_indices)):
            if hallucination_label["end"] < sentence_start_indices[i]:
                hallu_ends_in_sentence = i-1
                break
        else:
            hallu_ends_in_sentence = len(sentence_start_indices)-1

        # Overwriting the target with 1 for sentences that contain hallucinations
        for hallu_sent_i in range(hallu_starts_in_sentence, hallu_ends_in_sentence+1):
            sentence_targets[hallu_sent_i] = 1
    return sentence_targets

def read_responses_df(path: str) -> pd.DataFrame:
    # Read responses from json file and store it in a dataframe
    responses = []
    with open(path) as file:
        for line in file.read().split("\n"):
            if line.strip():
                responses.append(json.loads(line))
    return pd.DataFrame(responses)

def read_sources_df(path: str) -> pd.DataFrame:
    # Read source info from json file and store it in a dataframe
    sources = []
    with open(path) as file:
        for line in file.read().split("\n"):
            if line.strip():
                sources.append(json.loads(line))
    return pd.DataFrame(sources).set_index("source_id")

def get_all_sentences(models: list, cum=False, min_sent_length=20, invalid_starts_or_ends=":[](){}"):
    # Read responses from json file and store it in a dataframe
    responses = read_responses_df(os.path.join(RAGTRUTH_DIR, "response.jsonl"))

    # Read source info from json file and store it in a dataframe
    sources = read_sources_df(os.path.join(RAGTRUTH_DIR, "source_info.jsonl"))
    sources = sources[sources["task_type"] == "QA"]

    # Get only rows that contain responses written by the selected LLM
    model_responses = responses[(responses["model"].isin(models)) & (responses["quality"] == "good")].set_index("id")
    model_responses = model_responses.sort_values("source_id")

    all_sentences = []
    # Iterate over each response that has been generated by the model
    for model_name, model_responses_df in model_responses.groupby("model"):
        for response_id, response_row in model_responses_df.iterrows():
            if response_row["source_id"] not in sources.index:
                continue

            # Find prompt that is associated with this response
            source_row = sources.loc[response_row["source_id"]]
            prompt = source_row["prompt"]
            llm_response = response_row["response"]

            # Split response into sentences
            response_sentences = sentence_split(llm_response)

            # Get index of each sentence in the response
            sentence_start_indices = [llm_response.index(sent) for sent in response_sentences]

            # Get a list of ones and zeros (per sentence either 1 (= hallucination) or 0 (= no hallucination))
            targets = get_targets(sentence_start_indices, response_row["labels"])

            # List containing the cumulative concatenated sentences
            cum_sentences = cum_concat(llm_response, response_sentences, sentence_start_indices)

            for sent_i in range(len(response_sentences)):
                if cum:
                    sentence = cum_sentences[sent_i]
                else:
                    sentence = response_sentences[sent_i]

                if len(response_sentences[sent_i]) < min_sent_length or response_sentences[sent_i][0] in invalid_starts_or_ends or response_sentences[sent_i][-1] in invalid_starts_or_ends:
                    continue

                all_sentences.append({
                    "model": model_name,
                    "quantization": None,
                    "prompt": prompt,
                    "llm_response": llm_response,
                    "response_id": response_id,
                    "source_id": response_row["source_id"],
                    "target": targets[sent_i],
                    "sentence": sentence,
                    "question": source_row.loc["source_info"]["question"],
                    "passages": source_row.loc["source_info"]["passages"],
                })

    return pd.DataFrame(all_sentences)

def shd_classify_for_entire_llm_output(model_responses_df, model_i) -> list:
    data = []
    pbar = tqdm(total=SAMPLES_PER_LLM)
    for source_id, source_df in model_responses_df.sample(frac=1.0, replace=False, random_state=432+model_i).groupby("source_id"):
        row = source_df.iloc[-1]
        targets = source_df["target"].tolist()
        sentences = source_df["sentence"].tolist()

        if len(data) >= SAMPLES_PER_LLM:
            break

        d = json.loads(row.to_json())
        d["targets"] = targets
        d["sentences"] = sentences
        shd_response = classify(
            title="<not available>",
            section_before_passage="<not available>",
            passage_text=d["passages"],
            question=d["question"],
            answer_quote="<see section SENTENCE>",
            llm_output=d["llm_response"]
        )
        del d["sentence"]
        del d["target"]
        if shd_response["has_mistakes"] != len(shd_response["mistakes"]) > 0:
            pprint(shd_response)
            print("This cant be!")
            print("\n"*5)
            print(source_df)
            print("\n"*5)
            input("press to continue")

        predictions = [] #TODO
        for sent in sentences:
            sent_simplified = re.sub(SIMPLIFICATION_REGEX, '', sent.lower())
            label = "CORRECT"

            if shd_response["has_mistakes"]:
                for mistake in shd_response["mistakes"]:
                    mistake_quote_simplified = re.sub(SIMPLIFICATION_REGEX, '', mistake["output_quote"].lower())
                    if mistake_quote_simplified in sent_simplified:
                        label = mistake["type"]
                        break

            predictions.append(label)

        d["predictions"] = predictions
        print(targets)
        print(predictions)
        pbar.update()
    pbar.close()
    return data

def shd_classify_for_one_sentence(model_responses_df, model_i) -> list:
    hallu_df = model_responses_df[model_responses_df["target"] == 1].sample(SAMPLES_PER_LLM // 2, replace=False, random_state=432+model_i)
    grounded_df = model_responses_df[model_responses_df["target"] == 0].sample(SAMPLES_PER_LLM // 2, replace=False, random_state=432+model_i)

    model_responses_df = pd.concat([hallu_df, grounded_df], axis=0)

    data = []
    pbar = tqdm(total=SAMPLES_PER_LLM)
    for _, row in model_responses_df.iterrows():
        if len(data) >= SAMPLES_PER_LLM:
            break

        d = json.loads(row.to_json())
        shd_response = classify(
            title="<not available>",
            section_before_passage="<not available>",
            passage_text=d["passages"],
            question=d["question"],
            answer_quote="<see section SENTENCE>",
            llm_output=d["sentence"]
        )
        d["shd_response"] = shd_response
        if shd_response["has_mistakes"] != len(shd_response["mistakes"]) > 0:
            pprint(shd_response)
            print("This cant be!")
            print("\n"*5)
            print(row)
            print("\n"*5)
            input("press to continue")

        sent_simplified = re.sub(SIMPLIFICATION_REGEX, '', d["sentence"].lower())
        label = "CORRECT"

        try:
            if shd_response["has_mistakes"]:
                for mistake in shd_response["mistakes"]:
                    mistake_quote_simplified = re.sub(SIMPLIFICATION_REGEX, '', mistake["output_quote"].lower())
                    if mistake_quote_simplified in sent_simplified:
                        label = mistake["type"]
                        break
        except:
            label = "UNKNOWN"

        d["prediction"] = label
        print("target:", d["target"])
        print("label:", label)

        data.append(d)

        pbar.update()
    pbar.close()
    return data

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # data per LLM configuration (this will be filled)
    data = []
    
    all_sentences = get_all_sentences(['mistral-7B-instruct', 'llama-2-7b-chat', 'llama-2-13b-chat'])

    # Iterate over each response that has been generated by the model
    model_i = 0
    for model_name, model_responses_df in all_sentences.groupby("model"):
        data += shd_classify_for_one_sentence(model_responses_df, model_i)
        model_i += 1  

    save_to = os.path.join(CURR_DIR, f"benchmark_results.json")

    # Save data to json file
    print(f"Saving to '{save_to}'...")
    with open(save_to, 'w') as file:
        json.dump(data, file, indent=4)