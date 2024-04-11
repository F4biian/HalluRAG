# This should be called in the HalluRAG directory due to an import error that would occur otherwise.
# e.g.: python3 data/RAGTruth/internal_states_from_data.py

# TODO: Remove when publishing
import os
os.environ['HF_HOME'] = '/data2/quokka/.cache'

from models import Mistral_7B_Instruct_V1, LLaMA2_7B_ChatHF


llm = LLaMA2_7B_ChatHF(auto_load=True)

print(llm.generate("What is the capital of Austria, but spelled backwards?"))
# print(len(llm.get_internal_states("What is the capital of Austria, but spelled backwards?", 'The capital of Austria, spelled backwards, is "Austria".')))

llm.unload()