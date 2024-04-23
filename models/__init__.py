from .mistral import Mistral_7B_Instruct_V1
from .llama import LLaMA2_7B_ChatHF, LLaMA2_13B_ChatHF
from .gemma import Gemma_7B
from .llm import LLM

__all__ = [
    "Mistral_7B_Instruct_V1",
    "LLaMA2_7B_ChatHF",
    "LLaMA2_13B_ChatHF",
    "Gemma_7B",
    "set_hf_token",
]

def set_hf_token(token: str) -> None:
    LLM.hf_token = token