# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/RAGTruth/internal_states_from_data.py mistral-7B-instruct 42 512 ""


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

# Loading env variables (happens before importing models for the case if HF_HOME is changed)
load_dotenv()

# Importing models from own architecture
from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF, LLaMA2_13B_ChatHF, set_hf_token
from models.utils import sentence_split

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

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

def cum_concat(response, sentences, sentence_start_indices) -> List[str]:
    cum_sentences = []

    # Calculate the end index of each sentence
    sentence_end_indices = [sentence_start_indices[i] + len(sentences[i]) for i in range(len(sentence_start_indices))]

    for end_index in sentence_end_indices:
        cum_sentences.append(response[:end_index])

    return cum_sentences

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Either 'mistral-7B-instruct', 'llama-2-7b-chat' or 'llama-2-13b-chat'", type=str)
    parser.add_argument("sampling_seed", help="A unique integer for the random state when sampling data", type=int)
    parser.add_argument("response_count", help="Amount of responses that should be sampled from the data (max: 2965)", type=int, default=512)
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
    llm = llm_class(args.sampling_seed, args.quantization.strip() if args.quantization.strip() else None)

    # Number of responses that are sampled from an LLM (each LLM has 2965 responses)
    sample_response_count_for_each_llm = args.response_count

    # Load LLM into GPU
    llm.load()
    print(llm.model.config)
    print(llm.model)

    # Read responses from json file and store it in a dataframe
    responses = read_responses_df(os.path.join(CURR_DIR, "response.jsonl"))

    # Read source info from json file and store it in a dataframe
    sources = read_sources_df(os.path.join(CURR_DIR, "source_info.jsonl"))

    # Get only rows that contain responses written by the selected LLM
    model_responses = responses[(responses["model"] == model_name) & (responses["quality"] == "good")].set_index("id")

    # data per LLM configuration (this will be filled)
    data = []

    model_responses_shuffled = model_responses.sample(frac=1.0, replace=False, random_state=llm.sampling_seed)

    # Process bar in terminal
    pbar = tqdm(total=sample_response_count_for_each_llm, desc=str(llm))

    # Iterate over each response that has been generated by the model
    for response_id, response_row in model_responses_shuffled.iterrows():
        # Stop internal state extraction, once the given amount of responses have been processed
        if len(data) >= sample_response_count_for_each_llm:
            break

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
            "sentence_data": []
        }

        try:
            for sent_i in range(len(cum_sentences)):
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
            continue

        data.append(response_data)

        # Update process bar
        pbar.update()
        
    # Stop process bar
    pbar.refresh()
    pbar.close()

    # Save data to json file
    save_to = os.path.join(CURR_DIR, f"internal_states/{str(llm).replace('/', '_')}.json")
    print(f"Saving to '{save_to}'...")
    with open(save_to, "w") as file:
        json.dump(data, file, indent=4)