from models.llm import LLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
class LLaMA2_7B_ChatHF(LLM):
    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left", **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass

# https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
class LLaMA2_13B_ChatHF(LLM):
    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", padding_side="left", **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass
