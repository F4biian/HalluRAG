########################################################################################
# IMPORTS

import os
import warnings
import pandas as pd
import json
from typing import List
from tqdm import tqdm
from SHD.shd import classify
from pprint import pprint
import re
import argparse
import pickle
from data.wikipedia.analyze_articles import get_useful_articles
from difflib import SequenceMatcher
import numpy as np
from sklearn.utils import resample

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DATA_FOLDER = os.path.join(CURR_DIR, "..", "qna2output", "internal_states_new")
SIMPLIFICATION_REGEX = r'\s|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~'
BLACK_LIST = ["mistralai_Mistral-7B-Instruct-v0.1.pickle", "meta-llama_Llama-2-7b-chat-hf.pickle", "meta-llama_Llama-2-13b-chat-hf (int4).pickle"]

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

def get_all_sentences(filename: str):
    useful_articles = get_useful_articles()
    data = read_data(filename)["data"]

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
            "target": None, # This is determined later using SHD and IDKC
            "cum_sentence": llm_output,
            "internal_states": internal_states,
        }],
        "llm_response": llm_response
    }
    """

    all_sentences = []
    for d in data:
        for sent_i in range(len(d["sentence_data"])):
            sentence = d["sentence_data"][sent_i]["cum_sentence"]

            str_target = d["sentence_data"][sent_i]["target"]
            answerable = d["prompt"]["answerable"]
            
            # if answerable:
            #     if str_target == "W" or str_target == "IDK":
            #         target = 1
            #     elif str_target == "R":
            #         target = 0
            #     else:
            #         continue
            # else:
            #     if str_target == "W" or str_target == "IDK":
            #         target = 1
            #     elif str_target == "R":
            #         target = 0
            #     else:
            #         continue

            article_content = useful_articles[d["prompt"]["passage"]["useful_art_i"]]["content"]
            passage_start = d["prompt"]["passage"]["passage_start"]
            passage_end = d["prompt"]["passage"]["passage_end"]

            chunk = get_chunk_content(article_content, passage_start, passage_end, d["prompt"]["chunk_size"])

            all_sentences.append({
                "model": d["model"],
                "quantization": d["quantization"],
                "model_quantization": d["model"] + (f" ({d['quantization']})" if d["quantization"] else "") ,
                "answerable": answerable,
                "answer_chunk_index": d["prompt"]["answer_chunk_index"],
                "chunk_size": d["prompt"]["chunk_size"],
                "chunks_per_prompt": d["prompt"]["chunks_per_prompt"],
                "chunk": chunk,
                "uglified": d["prompt"]["uglified"],
                "prompt_template_name": d["prompt"]["prompt_template_name"],
                "passage": d["prompt"]["passage"],
                "other_passages": d["prompt"]["other_passages"],
                "llm_response": d["llm_response"],
                "target": str_target,
                "sentence": sentence,
                "question": d["prompt"]["passage"]["question"]
            })

    return pd.DataFrame(all_sentences)

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

def run(useful_articles, filename: str, is_test=False) -> None:
    save_to = os.path.join(CURR_DIR, "..", "HalluRAG", filename)
    data = []

    def save(use_indent=False):
        nonlocal data, save_to
        # Save data to json file
        # print(f"Saving to '{save_to}'...")
        with open(save_to, 'w') as file:
            if use_indent:
                json.dump(data, file, indent=4)
            else:
                json.dump(data, file)

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
            "target": None, # This is determined later using SHD and IDKC
            "cum_sentence": llm_output,
            "internal_states": internal_states,
        }],
        "llm_response": llm_response
    }
    """

    if is_test:
        answ = []
        answer_chunk_index = []
        chunk_size = []
        chunks_per_prompt = []
        prompt_template_name = []
        art_ids = []
        for d in responses:
            answ.append(d["prompt"]["answerable"])
            answer_chunk_index.append(d["prompt"]["answer_chunk_index"])
            chunk_size.append(d["prompt"]["chunk_size"])
            chunks_per_prompt.append(d["prompt"]["chunks_per_prompt"])
            prompt_template_name.append(d["prompt"]["prompt_template_name"])
            art_ids.append(d["prompt"]["passage"]["useful_art_i"])

        return {
            "len": len(responses),
            "answerable_mean": np.mean(answ),
            "answer_chunk_index": pd.Series(answer_chunk_index).value_counts().to_dict(),
            "chunk_size": pd.Series(chunk_size).value_counts().to_dict(),
            "chunks_per_prompt": pd.Series(chunks_per_prompt).value_counts().to_dict(),
            "prompt_template_name": pd.Series(prompt_template_name).value_counts().to_dict(),
        }, art_ids

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

        for section_i, section_name in enumerate(shd_response):
            section_dict = shd_response[section_name]
            # sent_i = int(section_name.split()[-1])-1

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

        # if input().lower().strip() == "x":
        #     exit()
    save(True)

def data_disbtr(responses):
    answ = []
    answer_chunk_index = []
    chunk_size = []
    chunks_per_prompt = []
    prompt_template_name = []
    for d in responses:
        answ.append(d["prompt"]["answerable"])
        answer_chunk_index.append(d["prompt"]["answer_chunk_index"])
        chunk_size.append(d["prompt"]["chunk_size"])
        chunks_per_prompt.append(d["prompt"]["chunks_per_prompt"])
        prompt_template_name.append(d["prompt"]["prompt_template_name"])

    return {
        "len": len(responses),
        "answerable_mean": np.mean(answ),
        "answer_chunk_index": pd.Series(answer_chunk_index).value_counts().to_dict(),
        "chunk_size": pd.Series(chunk_size).value_counts().to_dict(),
        "chunks_per_prompt": pd.Series(chunks_per_prompt).value_counts().to_dict(),
        "prompt_template_name": pd.Series(prompt_template_name).value_counts().to_dict()
    }

def oversample_responses(data, columns=["answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name"], max_iter=10000, no_improv_of=0.005, no_improv_after=100) -> pd.DataFrame:
    df_data = []

    for i, d in enumerate(data):
        prompt = d["prompt"]
        to_add = {"i": i}
        for key in prompt:
            if key in columns:
                to_add[key] = prompt[key]
        df_data.append(to_add)

    df = pd.DataFrame(df_data)
    orig_df = df.copy()

    unique_combos = df.drop_duplicates(columns, keep="first")
    combo_reproduced_amount = {}
    prev_delete_name = None

    def get_imbalance(df) -> float:
        err = 0
        for col in columns:
            # err += (df[col].value_counts() / df.shape[0]).std()
            rel_vs = (df[col].value_counts() / df.shape[0])
            err += (rel_vs.max() - rel_vs.min()) ** 2
        return err
    
    def find_best_combo_to_delete():
        nonlocal combo_reproduced_amount

        start_err = get_imbalance(df)
        best_ind = None
        best_err_delta = np.inf
        for ind, row in unique_combos.iterrows():
            combo_name = str(row[columns].to_dict())
            if combo_reproduced_amount.get(combo_name, 0) <= 0:
                continue
            if combo_name == str(df[columns].iloc[-1].to_dict()):
                continue
            combo_reproduced_amount[combo_name] = combo_reproduced_amount.get(combo_name, 0) + 1

            pot_df = df.drop(ind) # pd.concat([df, pd.DataFrame(row).T], ignore_index=True)
            new_err = get_imbalance(pot_df)
            err_delta = new_err - start_err
            if err_delta < best_err_delta:
                best_err_delta = err_delta
                best_ind = ind
        if best_ind is not None:
            return best_err_delta, unique_combos.loc[best_ind, columns].to_dict()
        return best_err_delta, None

    def find_best_combo():
        nonlocal prev_delete_name
        start_err = get_imbalance(df)
        best_ind = None
        best_err_delta = np.inf
        for ind, row in unique_combos.iterrows():
            if str(row[columns].to_dict()) == prev_delete_name:
                continue
            pot_df = df._append(row, ignore_index=True) # pd.concat([df, pd.DataFrame(row).T], ignore_index=True)
            new_err = get_imbalance(pot_df)
            err_delta = new_err - start_err
            if err_delta < best_err_delta:
                best_err_delta = err_delta
                best_ind = ind
        if best_ind is not None:
            return best_err_delta, unique_combos.loc[best_ind, columns].to_dict()
        return best_err_delta, None

    err = get_imbalance(df)
    max_iter_counter = 0
    err_hist = [err]
    while err > 0 and max_iter_counter < max_iter:
        best_err_delta, best_combo = find_best_combo()

        is_delete = False
        # d_best_err_delta, d_best_combo = find_best_combo_to_delete()
        # # print(d_best_err_delta)
        # is_delete = best_err_delta >= 0 or d_best_err_delta < 0 
        # if is_delete:
        #     best_combo = d_best_combo

        best_combo_str = str(best_combo)
        if best_combo_str not in combo_reproduced_amount:
            combo_reproduced_amount[best_combo_str] = 0

        combo_filter = np.ones(shape=df.shape[0], dtype=bool)
        combo_filter[:] = True
        for col in best_combo:
            combo_filter &= (df.loc[:, col] == best_combo[col])
        
        same_combos = df[combo_filter]
        if is_delete:
            prev_delete_name = best_combo_str
            min_ind = same_combos["i"].value_counts().index[0]
            # print("delete", (df["i"] == min_ind).index[-1], best_combo_str)
            df = df.drop((df["i"] == min_ind).index[-1])
            combo_reproduced_amount[best_combo_str] -= 1
        else:
            min_ind = same_combos["i"].value_counts().index[-1]
            # print("add", best_combo_str)
            row_to_reproduce = same_combos.loc[min_ind]
            # df = pd.concat([df, pd.DataFrame(row_to_reproduce).T], ignore_index=True)
            df = df._append(row_to_reproduce, ignore_index=True)
            combo_reproduced_amount[best_combo_str] += 1

        max_iter_counter += 1
        err = get_imbalance(df)
        err_hist.append(err)

        if len(err_hist) >= no_improv_after and err_hist[-no_improv_after] - err <= no_improv_of:
            break

        # print(err)

    best_i = np.argmin(err_hist) + orig_df.shape[0]
    df = df.iloc[:best_i]
    err = get_imbalance(df)

    new_data = []
    for i in df["i"]:
        new_data.append(data[i])

    return err, new_data

def load_data_balanced(model_name, to_balance=["answerable", "chunk_size", "chunks_per_prompt", "prompt_template_name"]):
    train_data = None
    val_data = None
    test_data = None
    for file in os.listdir(OUTPUT_DATA_FOLDER):
        typ, model = file.split("_", 1)
        if model != model_name:
            continue
        if typ == "train":
            train_data = read_data(os.path.join(OUTPUT_DATA_FOLDER, file))
        if typ == "val":
            val_data = read_data(os.path.join(OUTPUT_DATA_FOLDER, file))
        if typ == "test":
            test_data = read_data(os.path.join(OUTPUT_DATA_FOLDER, file))

    pprint(data_disbtr(train_data))
    err, train_data = oversample_responses(train_data, to_balance, 10000)
    pprint(data_disbtr(train_data))
    print(err)

def test():
    t = {}
    errs = {}
    l = 0

    common_art_ids = None
    for file in os.listdir(OUTPUT_DATA_FOLDER):
        is_black = False
        # for b in BLACK_LIST:
        #     if b.lower() in file.lower():
        #         is_black = True
        #         break
        # if is_black:
        #     continue
        # if "val" not in file:
        #     continue

        print(file)
        typ, na = file.split("_", 1)

        err, _ = (0, 0) # oversample_responses(read_data(os.path.join(OUTPUT_DATA_FOLDER, file)))

        if na not in t:
            t[na] = {}
            errs[na] = {}

        errs[na][typ] = err
        asd, art_ids = run(useful_articles, os.path.join(OUTPUT_DATA_FOLDER, file), True)
        t[na][typ] = asd
        l += t[na][typ]["len"]

        if common_art_ids is None:
            common_art_ids = art_ids
        else:
            new_common = []
            for id in art_ids:
                if id in common_art_ids:
                    new_common.append(id)
            common_art_ids = new_common

    for m in t:
        train = t[m]["train"]["len"]
        val = t[m]["val"]["len"]
        test = t[m]["test"]["len"]

        s = train + test + val

        print(m, f"{round(train / s, 2)}-{round(val / s, 2)}-{round(test / s, 2)}")

    pprint(t)
    print(l)
    pprint(errs)
    print("Common prompts:", len(common_art_ids))

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    useful_articles = get_useful_articles()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Pickle file to run SHD on", type=str)
    args = parser.parse_args()
    
    file = args.filename
    # test()
    # exit()
    run(useful_articles, file)