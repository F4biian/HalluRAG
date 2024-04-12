# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/RAGTruth/internal_states_from_data.py


########################################################################################
# IMPORTS

import os
import warnings
from dotenv import load_dotenv

# Loading env variables (happens before importing models for the case if HF_HOME is changed)
load_dotenv()

# Importing models from own architecture
from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF

########################################################################################


if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # with Mistral_7B_Instruct_V1() as llm:
    with LLaMA2_7B_ChatHF(os.environ['HF_TOKEN']) as llm:
        print(llm.generate("What is the capital of Austria, but spelled backwards?"))
        # print(len(llm.get_internal_states("What is the capital of Austria, but spelled backwards?", 'The capital of Austria, spelled backwards, is "Austria".')))