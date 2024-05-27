########################################################################################
# IMPORTS

import os
import warnings
import json
from dotenv import load_dotenv
from tqdm import tqdm

# Loading env variables (happens before importing models for the case if HF_HOME is changed)
load_dotenv()

# Importing models from own architecture
from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF, LLaMA2_13B_ChatHF, set_hf_token

########################################################################################

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_TO = os.path.join(CURR_DIR, "max_tokens.json")

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Get huggingface token from environment and set it for all llms
    set_hf_token(os.environ['HF_TOKEN'])

    all_models = [
        Mistral_7B_Instruct_V1(0, None),
        Mistral_7B_Instruct_V1(0, "float8"),
        Mistral_7B_Instruct_V1(0, "int8"),
        Mistral_7B_Instruct_V1(0, "int4"),
        LLaMA2_7B_ChatHF(0, None),
        LLaMA2_7B_ChatHF(0, "float8"),
        LLaMA2_7B_ChatHF(0, "int8"),
        LLaMA2_7B_ChatHF(0, "int4"),
        LLaMA2_13B_ChatHF(0, None),
        LLaMA2_13B_ChatHF(0, "float8"),
        LLaMA2_13B_ChatHF(0, "int8"),
        LLaMA2_13B_ChatHF(0, "int4"),
    ]

    data = {}

    for llm in tqdm(all_models):
        llm.load()
        data[str(llm)] = llm.get_max_token_count_on_gpu()
        llm.unload()

    # Save data to json file
    print(f"Saving to '{SAVE_TO}'...")
    with open(SAVE_TO, 'w') as file:
        json.dump(data, file, indent=4)