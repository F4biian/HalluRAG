from models.llm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
class Mistral_7B_Instruct_V1(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side="left", **self.tokenizer_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extend_generation_config(self, generation_config: dict) -> None:
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id
