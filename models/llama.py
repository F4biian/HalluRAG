from models.llm import LLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
class LLaMA2_7B_ChatHF(LLM):
    def __init__(self, huggingface_token: str, default_temperature: float = 0, auto_load: bool = False) -> None:
        self.huggingface_token = huggingface_token
        super().__init__(default_temperature, auto_load)
        
    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", token=self.huggingface_token)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left", token=self.huggingface_token)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass
