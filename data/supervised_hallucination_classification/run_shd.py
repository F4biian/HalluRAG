# Exmaple to run this script:
# python3 data/supervised_hallucination_classification/run_shd.py "train_mistralai_Mistral-7B-Instruct-v0.1.pickle"

########################################################################################
# IMPORTS

import os
import warnings
from typing import List
from tqdm import tqdm
from SHD.shd import classify
import re
import argparse
import pickle
from data.wikipedia.analyze_articles import get_useful_articles
from difflib import SequenceMatcher

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DATA_FOLDER = os.path.join(CURR_DIR, "..", "qna2output", "internal_states")
SAVE_TO = os.path.join(CURR_DIR, "..", "HalluRAG")
SIMPLIFICATION_REGEX = r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~'
BLACK_LIST = []

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

def read_data(filename: str) -> list:
    with open(os.path.join(OUTPUT_DATA_FOLDER, filename), 'rb') as handle:
        return pickle.load(handle)

def get_chunk_content(article_content: str, passage_start: int, passage_end: int, chunk_size: int) -> str:
    chars_left = max(chunk_size - (passage_end - passage_start), 0)

    chunk_start = passage_start
    chunk_end   = passage_end

    if chars_left > 0:
        chunk_start = max(chunk_start-chunk_size, 0)
        chunk_end   = min(chunk_end+chunk_size, len(article_content))

    chunk = article_content[chunk_start:chunk_end]
    
    # Replace multiple consecutive newlines with a single newline
    chunk = re.sub(r'\n+', '\n', chunk)

    return chunk

def isin(part: str, whole: str) -> bool:
    p = re.sub(SIMPLIFICATION_REGEX, '', part.lower())
    w = re.sub(SIMPLIFICATION_REGEX, '', whole.lower())
    return p in w

def is_similar(a: str, b: str, thr: float=0.9) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= thr

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
                    # print("-"*10)
                    # pprint(pred["llm_eval"])
                    # print("-"*10)
                    return None
                    # raise Exception("Unanswerable question has been answered.")
                else:
                    prediction = 1
            else:
                prediction = 0
    return prediction

def run(useful_articles, filename: str) -> None:
    if not os.path.exists(SAVE_TO):
        os.makedirs(SAVE_TO)

    save_to = os.path.join(SAVE_TO, filename)
    data = []

    def save():
        nonlocal data, save_to
        # Save data to pickle file
        with open(save_to, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    responses = read_data(filename)

    data.clear()

    """
    {
        "model": llm.name,
        "quantization": llm.quantization,
        "prompt": {
            "answerable": True,
            "answer_chunk_index": answer_chunk_index,
            "chunk_size": chunk_size,
            "chunks_per_prompt": chunks_per_prompt,
            "uglified": uglify_bool,
            "prompt_template_name": prompt_template_name,
            "passage": row.to_dict(),
            "other_passages": [other_chunk_row.to_dict() for _, other_chunk_row in other_chunks_df.iterrows()],
            "rag_prompt": prompt_template_function(chunks_for_answerable, row["question"])
        },
        "sentence_data": [{
            "target": None, # This is determined later using SHD
            "cum_sentence": llm_output,
            "internal_states": internal_states,
        }],
        "llm_response": llm_response
    }
    """

    for d in tqdm(responses):
        answerable = d["prompt"]["answerable"]
        article_content = useful_articles[d["prompt"]["passage"]["useful_art_i"]]["content"]
        passage_start = d["prompt"]["passage"]["passage_start"]
        passage_end = d["prompt"]["passage"]["passage_end"]
        chunk = get_chunk_content(article_content, passage_start, passage_end, d["prompt"]["chunk_size"])

        # print(d["prompt"]["answerable"])
        # print(d["model"])
        # print(d["quantization"])

        llm_output_split = []
        predictions = []
        targets = []
        last_sent_i = 0
        for sent_d in d["sentence_data"]:
            cum_sent = sent_d["cum_sentence"]
            llm_output_split.append(cum_sent[last_sent_i:].strip())
            last_sent_i = len(cum_sent)
            predictions.append(None)
            targets.append(None)

        try:
            shd_response = classify(
                title=d["prompt"]["passage"]["article_title"],
                chunk=chunk,
                chunk_index=d["prompt"]["answer_chunk_index"],
                question=d["prompt"]["passage"]["question"],
                answer_quote=d["prompt"]["passage"]["answer_quote"],
                llm_output_split=llm_output_split,
                titles=[pas["article_title"] for pas in d["prompt"]["other_passages"]],
                answerable=answerable,
                verbose=False
            )
        except KeyboardInterrupt:
            print("Stopping...")
            save()
            exit()
        except:
            shd_response = {}

        for section_i, section_name in enumerate(shd_response):
            section_dict = shd_response[section_name]

            p_dict = {}

            try:
                p_dict["conflicting_fail_content"] = not is_similar(llm_output_split[section_i], section_dict["conflicting"]["section_content"], 0.95)
            except:
                p_dict["conflicting_fail_content"] = None
            try:
                p_dict["conflicting_fail"] = (not isin(section_dict["conflicting"]["necessary_chunk_quote"], chunk)) or (not isin(section_dict["conflicting"]["section_quote"], llm_output_split[section_i]))
            except:
                p_dict["conflicting_fail"] = None

            try:
                p_dict["grounded_fail_content"] = not is_similar(llm_output_split[section_i], section_dict["grounded"]["section_content"], 0.95)
            except:
                p_dict["grounded_fail_content"] = None
            try:
                p_dict["grounded_fail"] = (not isin(section_dict["grounded"]["necessary_chunk_quote"], chunk)) or (not isin(section_dict["grounded"]["section_quote"], llm_output_split[section_i]))
            except:
                p_dict["grounded_fail"] = None

            try:
                p_dict["no_clear_answer_fail_content"] = not is_similar(llm_output_split[section_i], section_dict["cannot_really_answer"]["section_content"], 0.95)
            except:
                p_dict["no_clear_answer_fail_content"] = None
            try:
                p_dict["no_clear_answer_fail"] = not isin(section_dict["cannot_really_answer"]["section_quote"], llm_output_split[section_i])
            except:
                p_dict["no_clear_answer_fail"] = None

            try:
                p_dict["conflicting"] = section_dict["conflicting"]["result"]
            except:
                p_dict["conflicting"] = None
            try:
                p_dict["grounded"] = section_dict["grounded"]["result"]
            except:
                p_dict["grounded"] = None
            try:
                p_dict["has_factual_information"] = section_dict["grounded"]["has_factual_information"]
            except:
                p_dict["has_factual_information"] = None
            try:
                p_dict["no_clear_answer"] = section_dict["cannot_really_answer"]["result"]
            except:
                p_dict["no_clear_answer"] = None
            try:
                p_dict["llm_eval"] = section_dict
            except:
                p_dict["llm_eval"] = None

            if section_i < len(predictions):
                predictions[section_i] = p_dict

            targets.append(get_shd_prediction(answerable, p_dict))

        for pred_i, pred in enumerate(predictions):
            if pred_i >= len(d["sentence_data"]):
                continue
            d["sentence_data"][pred_i]["pred"] = pred
            d["sentence_data"][pred_i]["target"] = targets[pred_i]
            
        data.append(d)
        save()
    save()

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    useful_articles = get_useful_articles()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Pickle file to run SHD on", type=str)
    args = parser.parse_args()
    
    file = args.filename
    run(useful_articles, file)