########################################################################################
# IMPORTS

import os
import warnings
import pandas as pd
import json
from typing import List
from tqdm import tqdm
from models.utils import sentence_split, cum_concat
from SHD.shd import classify, classify_unsupervised
from pprint import pprint
import re
from data.wikipedia.analyze_articles import get_useful_articles
from difflib import SequenceMatcher

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MANUAL_FILE = os.path.join(CURR_DIR, "manual_classification.json")
SAMPLES_PER_LLM = 50 # 3 llms
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

def read_data(filename: str) -> list:
    with open(filename) as file:
        data = json.load(file)
        return data

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

def get_all_sentences():
    useful_articles = get_useful_articles()
    data = read_data(MANUAL_FILE)["data"]

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

def shd_classify_for_entire_llm_output(model_responses_df, model_i) -> list:
    data = []
    pbar = tqdm(total=SAMPLES_PER_LLM)
    row = source_df.iloc[-1]
    for _, row in model_responses_df.iterrows():
        targets = row["target"]
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
    data = []
    pbar = tqdm(total=model_responses_df.shape[0])
    for _, row in model_responses_df.iterrows():
        d = json.loads(row.to_json())

        print("answerable:", d["answerable"])
        print("model:", d["model"])
        print("quant:", d["quantization"])
        shd_response = classify(
            title=d["passage"]["article_title"],
            chunk=d["chunk"],
            chunk_index=d["answer_chunk_index"],
            question=d["question"],
            answer_quote=d["passage"]["answer_quote"],
            llm_output=d["sentence"],
            titles=[pas["article_title"] for pas in d["other_passages"]],
            answerable=d["answerable"]
        )
        print(shd_response)
        input()

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

def isin(part: str, whole: str) -> bool:
    p = re.sub(SIMPLIFICATION_REGEX, '', part.lower())
    w = re.sub(SIMPLIFICATION_REGEX, '', whole.lower())
    return p in w

def is_similar(a: str, b: str, thr: float=0.9) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= thr

save_to = os.path.join(CURR_DIR, f"manual_benchmark_results.json")
data = []

def save(use_indent=False):
    global data, save_to
    # Save data to json file
    # print(f"Saving to '{save_to}'...")
    with open(save_to, 'w') as file:
        if use_indent:
            json.dump(data, file, indent=4)
        else:
            json.dump(data, file)

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    useful_articles = get_useful_articles()
    responses = read_data(MANUAL_FILE)["data"]

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
            answerable=d["prompt"]["answerable"],
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

        for pred_i, pred in enumerate(predictions):
            if pred_i >= len(d["sentence_data"]):
                continue
            d["sentence_data"][pred_i]["pred"] = pred

        data.append(d)
        save()

        # if input().lower().strip() == "x":
        #     exit()
    save(True)