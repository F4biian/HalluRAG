# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/RAGTruth/internal_states_from_data.py mistral-7B-instruct 512 ""


########################################################################################
# IMPORTS

import os
import warnings
import pandas as pd
import json
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm
import argparse
import pickle

# Loading env variables (happens before importing models for the case if HF_HOME is changed)
load_dotenv()

# Importing models from own architecture
from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF, LLaMA2_13B_ChatHF, set_hf_token
from models.utils import sentence_split, cum_concat

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_IDS_FILE = os.path.join(CURR_DIR, "common_source_ids.json")

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

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Either 'mistral-7B-instruct', 'llama-2-7b-chat' or 'llama-2-13b-chat'", type=str)
    parser.add_argument("response_count", help="Amount of responses that should be sampled from the data (max: -1)", type=int, default=512)
    parser.add_argument("quantization", help="Either None, 'float8', 'int8' or 'int4'", type=str, default=None)
    args = parser.parse_args()

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Get huggingface token from environment and set it for all llms
    set_hf_token(os.environ['HF_TOKEN'])

    # Select the LLM configuration that should be used
    model_name = args.model_name
    llm_class = None
    if model_name == "mistral-7B-instruct":
        llm_class = Mistral_7B_Instruct_V1
    elif model_name == "llama-2-7b-chat":
        llm_class = LLaMA2_7B_ChatHF
    elif model_name == "llama-2-13b-chat":
        llm_class = LLaMA2_13B_ChatHF
    llm = llm_class(0, args.quantization.strip() if args.quantization.strip() else None)

    # File containing the results at the end
    save_to = os.path.join(CURR_DIR, f"internal_states/{str(llm).replace('/', '_')}.pickle") # f"long_prompt_internal_states/{str(llm).replace('/', '_')}.pickle"

    # Load LLM into GPU
    llm.load()
    print(llm.model.config)
    print(llm.model)

    # Read responses from json file and store it in a dataframe
    responses = read_responses_df(os.path.join(CURR_DIR, "response.jsonl"))

    # Read source info from json file and store it in a dataframe
    sources = read_sources_df(os.path.join(CURR_DIR, "source_info.jsonl"))
    sources = sources[sources["task_type"] == "QA"]

    # Get only rows that contain responses written by the selected LLM
    model_responses = responses[(responses["model"] == model_name) & (responses["quality"] == "good")].set_index("id")
    model_responses = model_responses.sort_values("source_id")

    # data per LLM configuration (this will be filled)
    data = []

    # Number of responses that are taken from an LLM (each LLM has max. 2965 responses)
    if args.response_count > 0:
        response_count_for_each_llm = args.response_count
    else:
        response_count_for_each_llm = model_responses.shape[0]

    # Load list of source ids whose RAGTruth prompts (+each LLM answer) fits into the GPU
    with open(SOURCE_IDS_FILE, "r") as file:
        common_source_ids = json.loads(file.read())

    # Process bar in terminal
    pbar = tqdm(total=response_count_for_each_llm, desc=str(llm))

    # Used for long prompt internal states
    # sources["len"] = sources["prompt"].apply(len)
    # model_responses["len"] = model_responses["source_id"].apply(lambda src_id: sources.loc[src_id]["len"]) + model_responses["response"].apply(len)
    # model_responses = model_responses.sort_values("len", ascending=False)
    # print(model_responses[~model_responses["source_id"].isin(common_source_ids[:750])]) # actually without ":750": 184 rows

    # Iterate over each response that has been generated by the model
    for response_id, response_row in model_responses.iterrows():
        # Skip responses that are not associated with QA tasks
        if response_row["source_id"] not in sources.index:
            continue

        # Stop internal state extraction, once the given amount of responses have been processed
        if len(data) >= response_count_for_each_llm and args.response_count > 0:
            break

        # Skip sources that do not fit into GPU for at least one LLM
        if response_row["source_id"] not in common_source_ids: # When long prompt internal states: replace "not in" with "in"
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

        response_data = {
            "model": llm.name,
            "quantization": llm.quantization,
            "prompt": prompt,
            "sentence_data": [],
            "response_id": response_id,
            "source_id": response_row["source_id"]
        }

        try:
            for sent_i in range(len(cum_sentences)):
                # ignore if sentence current sentence length is below 10 chars TODO 
                asd
                llm_output = cum_sentences[sent_i]
                internal_states = llm.get_internal_states(prompt=prompt, llm_output=llm_output)

                response_data["sentence_data"].append({
                    "target": targets[sent_i],
                    "cum_sentence": llm_output,
                    "internal_states": internal_states,
                })
        except RuntimeError as err:
            # Skip iterations when cuda is complaining about too little memory
            print(err)
            if args.response_count < 0:
                pbar.update()
            continue

        data.append(response_data)

        # Update process bar
        pbar.update()
        
    # Stop process bar
    pbar.refresh()
    pbar.close()

    # Save data to pickle file
    print(f"Saving to '{save_to}'...")
    with open(save_to, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)