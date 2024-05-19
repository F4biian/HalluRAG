# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/qna2output/2output.py mistral-7B-instruct -1 "" train

########################################################################################
# IMPORTS

from dotenv import load_dotenv
import json
import os
import traceback
import datetime
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import argparse
import pickle
import warnings
import random
import re
from rag_prompts import CHUNK_SIZE, CHUNKS_PER_PROMPT, PROMPT_TEMPLATES, UGLIFY, uglify
from data.wikipedia.analyze_articles import get_useful_articles

# Loading env variables
load_dotenv()

# Importing models from own architecture
from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF, LLaMA2_13B_ChatHF, Gemma_7B_Instruct, set_hf_token
from models.utils import sentence_split, cum_concat

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
QNA_FILE = os.path.join(os.path.join(os.path.join(CURR_DIR, ".."), "wikipedia2qna"), "qna_per_passage.json")
OUTPUT_FILE = os.path.join(CURR_DIR, "rag_output.json")
LOG_FILE = os.path.join(CURR_DIR, "log.log")
RANDOM_STATE = 432
TRAIN_PERC = 0.75
VAL_PERC = 0.15 # also implying a TEST_PERC of 0.15

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def log(msg: str) -> None:
    with open(LOG_FILE, "a") as file:
        file.write(f"[{datetime.datetime.utcnow()}] {msg}\n")

def filter_qna_df(qna_df: pd.DataFrame) -> pd.DataFrame:
    qna_df_new = []

    # Pairs the starting word of a question with its overall amount of occurences in the dataset
    # This is for a better balance of different question types. For instance, we do not want
    # a lot of "when" questions if we could have less of those (due to removing qna pairs that
    # started differently).
    starting_words_counter = {}

    # Group qnas by their article
    article_qnas = qna_df.groupby("useful_art_i")

    for useful_art_i, article_qna_df in article_qnas:
        article_qna_df["first_word"] = article_qna_df["question"].apply(lambda q: q.split()[0].lower())
        article_qna_df["first_word_occurences"] = article_qna_df["first_word"].apply(lambda word: starting_words_counter[word] if word in starting_words_counter else 0)
        row_to_take = article_qna_df.sort_values("first_word_occurences").iloc[0]
        if row_to_take["first_word"] in starting_words_counter:
            starting_words_counter[row_to_take["first_word"]] += 1
        else:
            starting_words_counter[row_to_take["first_word"]] = 1
        
        qna_df_new.append(row_to_take.copy())

    qna_df_new = pd.DataFrame(qna_df_new).drop(["first_word", "first_word_occurences"], axis=1)

    return qna_df_new

def get_chunk_content(article_content: str, passage_start: int, passage_end: int, chunk_size: int) -> str:
    chars_left = max(chunk_size - (passage_end - passage_start), 0)

    chunk_start = passage_start
    chunk_end   = passage_end

    if chars_left > 0:
        chars_before = random.randint(0, chars_left)
        chars_after  = chars_left - chars_before
        chunk_start = max(chunk_start-chars_before, 0)
        chunk_end   = min(chunk_end+chars_after, len(article_content))

    chunk = article_content[chunk_start:chunk_end]
    
    # Replace multiple consecutive newlines with a single newline
    chunk = re.sub(r'\n+', '\n', chunk)

    return chunk

def get_rag_prompts(qna_df: pd.DataFrame, useful_articles: list) -> List[Dict[str, Any]]:
    prompts = []

    # Create a list with all parameters in equal balance (e.g. 1/3 of -1s, 1/3 of 250s and 1/3 of 500s)
    chunk_sizes = []
    chunks_per_prompts = []
    prompt_templates = []
    uglify_bools = []
    for i in range(qna_df.shape[0]):
        chunk_sizes.append(CHUNK_SIZE[i % len(CHUNK_SIZE)])
        chunks_per_prompts.append(CHUNKS_PER_PROMPT[i % len(CHUNKS_PER_PROMPT)])
        prompt_templates.append(PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)])
        uglify_bools.append(UGLIFY[i % len(UGLIFY)])

    # Shuffle the first three lists/arrays (the third does not need to be shuffled)
    chunk_sizes = np.array(chunk_sizes, dtype=int)
    chunks_per_prompts = np.array(chunks_per_prompts, dtype=int)
    uglify_bools = np.array(uglify_bools, dtype=bool)
    np.random.shuffle(chunk_sizes)
    np.random.shuffle(chunks_per_prompts)
    np.random.shuffle(uglify_bools)

    for i, (index, row) in enumerate(qna_df.iterrows()):
        chunk_size = int(chunk_sizes[i])
        chunks_per_prompt = int(chunks_per_prompts[i])
        uglify_bool = uglify_bools[i]
        prompt_template_function = prompt_templates[i]
        prompt_template_name = prompt_template_function.__name__

        article_content = useful_articles[row["useful_art_i"]]["content"]
        content = get_chunk_content(article_content, row["passage_start"], row["passage_end"], chunk_size)
        if uglify_bool:
            current_chunk = {
                "title": row["article_title"],
                "content": uglify(content)
            }
        else:
            current_chunk = {
                "title": row["article_title"],
                "content": content
            }

        # Get <chunks_per_prompt> other chunks (that are not from the same article)
        other_qna_articles = qna_df[qna_df["useful_art_i"] != row["useful_art_i"]]
        other_chunks_df = other_qna_articles.sample(n=chunks_per_prompt, random_state=RANDOM_STATE)

        other_chunks = []
        for _, other_chunk_row in other_chunks_df.iterrows():
            article_content = useful_articles[other_chunk_row["useful_art_i"]]["content"]
            content = get_chunk_content(article_content, other_chunk_row["passage_start"], other_chunk_row["passage_end"], chunk_size)
            if uglify_bool:
                other_chunks.append({
                    "title": other_chunk_row["article_title"],
                    "content": uglify(content)
                })
            else:
                other_chunks.append({
                    "title": other_chunk_row["article_title"],
                    "content": content
                })

        # This is the index of the chunk in the RAG prompt of the answerable question      
        answer_chunk_index = np.random.randint(0, chunks_per_prompt)

        # Add the RAG prompt with the unanswerable question 
        prompts.append({
            "answerable": False,
            "answer_chunk_index": None,
            "chunk_size": chunk_size,
            "chunks_per_prompt": chunks_per_prompt,
            "uglified": uglify_bool,
            "prompt_template_name": prompt_template_name,
            "passage": row.to_dict(),
            "other_passages": [other_chunk_row.to_dict() for _, other_chunk_row in other_chunks_df.iterrows()],
            "rag_prompt": prompt_template_function(other_chunks, row["question"])
        })

        # Replace the "wrong" chunk at <answer_chunk_index> with the answer chunk
        chunks_for_answerable = other_chunks.copy()
        chunks_for_answerable[answer_chunk_index] = current_chunk

        # Add the RAG prompt with the answerable question
        prompts.append({
            "answerable": True,
            "answer_chunk_index": answer_chunk_index,
            "chunk_size": chunk_size,
            "chunks_per_prompt": chunks_per_prompt,
            "uglified": uglify_bool,
            "prompt_template_name": prompt_template_name,
            "passage": row.to_dict(),
            "other_passages": [other_chunk_row.to_dict() for _, other_chunk_row in other_chunks_df.iterrows()],
            "rag_prompt": prompt_template_function(chunks_for_answerable, row["question"])
        })

    return prompts

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Either 'gemma-7b-it', 'mistral-7B-instruct', 'llama-2-7b-chat' or 'llama-2-13b-chat'", type=str)
    parser.add_argument("response_count", help="Amount of responses to generate", type=int, default=-1)
    parser.add_argument("quantization", help="Either None, 'float8', 'int8' or 'int4'", type=str, default=None)
    parser.add_argument("dataset", help="Either 'train', 'val', or 'test'", type=str, default="train")
    args = parser.parse_args()

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Get huggingface token from environment and set it for all llms
    set_hf_token(os.environ['HF_TOKEN'])

    useful_articles = get_useful_articles()
    with open(QNA_FILE, "r") as file:
        qna_df = pd.DataFrame(json.load(file))
    qna_df = filter_qna_df(qna_df)

    all_prompts = get_rag_prompts(qna_df, useful_articles)

    # Take those that are determined for the dataset type (train/val/test)
    if args.dataset.strip().lower() == "train":
        all_prompts = all_prompts[:int(len(all_prompts) * TRAIN_PERC)]
    elif args.dataset.strip().lower() == "val":
        all_prompts = all_prompts[int(len(all_prompts) * TRAIN_PERC):int(len(all_prompts) * (TRAIN_PERC + VAL_PERC))]
    elif args.dataset.strip().lower() == "test":
        all_prompts = all_prompts[int(len(all_prompts) * (TRAIN_PERC + VAL_PERC)):]
    else:
        raise Exception(f"Unknown dataset type '{args.dataset}'. Please choose one of 'train', 'val', or 'test'!")

    log(f"Length of all_prompts: {len(all_prompts)}")

    # Select the LLM configuration that should be used
    model_name = args.model_name
    llm_class = None
    if model_name == "mistral-7B-instruct":
        llm_class = Mistral_7B_Instruct_V1
    elif model_name == "llama-2-7b-chat":
        llm_class = LLaMA2_7B_ChatHF
    elif model_name == "llama-2-13b-chat":
        llm_class = LLaMA2_13B_ChatHF
    elif model_name == "gemma-7b-it":
        llm_class = Gemma_7B_Instruct
    llm = llm_class(0, args.quantization.strip() if args.quantization.strip() else None)

    # File containing the results at the end
    save_to = os.path.join(CURR_DIR, f"internal_states/{args.dataset}_{str(llm).replace('/', '_')}.pickle")

    # Load LLM into GPU
    llm.load()
    log(llm.model.config)
    log(llm.model)

    if args.response_count > 0:
        response_count_max = min(args.response_count, len(all_prompts))
    else:
        response_count_max = len(all_prompts)

    # Process bar in terminal
    pbar = tqdm(total=response_count_max, desc=str(llm))

    data = []

    # Now, give each prompt to LLM and get answer
    for prompt_i in range(len(all_prompts)):
        # Stop generation, once the given amount of responses has been made
        if len(data) >= response_count_max:
            break

        prompt_dict = all_prompts[prompt_i]
        rag_prompt = prompt_dict["rag_prompt"]

        # Delete rag_prompt from prompt_dict in order to keep response_data smaller
        # (rag_prompt is also redundant)
        del prompt_dict["rag_prompt"]

        if len(rag_prompt) == 1:
            system = None
            prompt = rag_prompt[0]["content"]
        else:
            system = rag_prompt[0]["content"]
            prompt = rag_prompt[1]["content"]

        llm_response = llm.generate(prompt=prompt, system=system)

        # Split response into sentences
        response_sentences = sentence_split(llm_response)

        # Get index of each sentence in the response
        sentence_start_indices = [llm_response.index(sent) for sent in response_sentences]

        # List containing the cumulative concatenated sentences
        cum_sentences = cum_concat(llm_response, response_sentences, sentence_start_indices)

        response_data = {
            "model": llm.name,
            "quantization": llm.quantization,
            "prompt": prompt_dict,
            "sentence_data": [],
            "llm_response": llm_response
        }

        try:
            for sent_i in range(len(cum_sentences)):
                llm_output = cum_sentences[sent_i]
                internal_states = llm.get_internal_states(prompt=prompt, llm_output=llm_output)

                response_data["sentence_data"].append({
                    "target": None, # This is determined later using SHD and IDKC
                    "cum_sentence": llm_output,
                    "internal_states": internal_states,
                })
        except RuntimeError as err:
            # Skip iterations when cuda is complaining about too little memory
            log(err)
            continue

        data.append(response_data)

        # Update process bar
        pbar.update()
        
    # Stop process bar
    pbar.refresh()
    pbar.close()

    # Save data to pickle file
    log(f"Saving to '{save_to}'...")
    with open(save_to, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)