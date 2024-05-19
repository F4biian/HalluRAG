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
from rag_prompts import CHUNK_SIZE, CHUNKS_PER_PROMPT, PROMPT_TEMPLATES

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
QNA_FILE = os.path.join(os.path.join(os.path.join(CURR_DIR, ".."), "wikipedia2qna"), "qna_per_passage.json")
OUTPUT_FILE = os.path.join(CURR_DIR, "rag_output.json")
LOG_FILE = os.path.join(CURR_DIR, "log.log")
RANDOM_STATE = 432

# Loading env variables
load_dotenv()

np.random.seed(RANDOM_STATE)

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

def get_chunk_content(context: str, passage: str, chunk_size: int = -1) -> str:
    entire_chunk = f"{context} {passage}"

    # The chunk be only truncated at the beginning in order to ensure that if a question
    # is considered answerable the answer still needs to be contained in the chunk (passage).
    if chunk_size > 0:
        entire_chunk[-chunk_size:]
    return entire_chunk

def get_rag_prompts(qna_df: pd.DataFrame) -> List[Dict[str, Any]]:
    prompts = []

    # Create a list with all parameters in equal balance (e.g. 1/3 of -1s, 1/3 of 250s and 1/3 of 500s)
    chunk_sizes = []
    chunks_per_prompts = []
    prompt_templates = []
    for i in range(qna_df.shape[0]):
        chunk_sizes.append(CHUNK_SIZE[i % len(CHUNK_SIZE)])
        chunks_per_prompts.append(CHUNKS_PER_PROMPT[i % len(CHUNKS_PER_PROMPT)])
        prompt_templates.append(PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)])

    # Shuffle the first two lists/arrays (the third does not need to be shuffled)
    chunk_sizes = np.array(chunk_sizes, dtype=int)
    chunks_per_prompts = np.array(chunks_per_prompts, dtype=int)
    np.random.shuffle(chunk_sizes)
    np.random.shuffle(chunks_per_prompts)

    for i, (index, row) in enumerate(qna_df.iterrows()):
        chunk_size = int(chunk_sizes[i])
        chunks_per_prompt = int(chunks_per_prompts[i])
        prompt_template_function = prompt_templates[i]
        prompt_template_name = prompt_template_function.__name__

        current_chunk = {
            "title": row["article_title"],
            "content": get_chunk_content(row["context"], row["passage_text"], chunk_size)
        }

        # Get <chunks_per_prompt> other chunks (that are not from the same article)
        other_qna_articles = qna_df[qna_df["useful_art_i"] != row["useful_art_i"]]
        other_chunks_df = other_qna_articles.sample(n=chunks_per_prompt, random_state=RANDOM_STATE)

        other_chunks = []
        for _, other_chunk_row in other_chunks_df.iterrows():
             other_chunks.append({
            "title": other_chunk_row["article_title"],
            "content": get_chunk_content(other_chunk_row["context"], other_chunk_row["passage_text"], chunk_size)
        })

        # This is the index of the chunk in the RAG prompt of the answerable question      
        answer_chunk_index = np.random.randint(0, chunks_per_prompt)

        # Add the RAG prompt with the unanswerable question 
        prompts.append({
            "answerable": False,
            "answer_chunk_index": None,
            "chunk_size": chunk_size,
            "chunks_per_prompt": chunks_per_prompt,
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
            "prompt_template_name": prompt_template_name,
            "passage": row.to_dict(),
            "other_passages": [other_chunk_row.to_dict() for _, other_chunk_row in other_chunks_df.iterrows()],
            "rag_prompt": prompt_template_function(chunks_for_answerable, row["question"])
        })

    return prompts

if __name__ == "__main__":
    with open(QNA_FILE, "r") as file:
        qna_df = pd.DataFrame(json.load(file))
    
    qna_df = filter_qna_df(qna_df)

    prompts = get_rag_prompts(qna_df)

    # TODO: generate LLM responses