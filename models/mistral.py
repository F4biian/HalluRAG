from models.llm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
class Mistral_7B_Instruct_V1(LLM):
    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", padding_side="left", **self.tokenizer_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extend_generation_config(self, generation_config: dict) -> None:
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
