from models.llm import LLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
class LLaMA2_7B_ChatHF(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("meta-llama/Llama-2-7b-chat-hf", quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.name, **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass

# https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
class LLaMA2_13B_ChatHF(LLM):
    def __init__(self, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("meta-llama/Llama-2-13b-chat-hf", quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.name, **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass
