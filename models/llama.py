from models.llm import LLM
from transformers import LlamaForCausalLM, LlamaTokenizer

# For activation value extraction do the following:
# Go to the transformers package on your local disk (e.g. ~/.local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py)
# In line 240 (version: transformers @ git+https://github.com/huggingface/transformers.git@8127f39624f587bdb04d55ab655df1753de7720a)
# replace the inner else block section with this:
'''
# Original version
# down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Version taken from MIND (https://github.com/oneal2000/MIND/issues/2):
a = self.act_fn(self.gate_proj(x))
self.activation_values_from_inserted_code = a.clone().detach()
down_proj = self.down_proj(a * self.up_proj(x))
'''

# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
class LLaMA2_7B_ChatHF(LLM):
    def __init__(self, sampling_seed: int, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("meta-llama/Llama-2-7b-chat-hf", sampling_seed, quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.name, **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass

    # def extract_internal_states_from_output(self, output) -> Dict[str, Any]:
    #     """
    #     Shapes:
    #         - output.logits: [torch.Size([1, <sequence length>, 32000])]
    #         - output.hidden_states: [33, torch.Size([1,  <sequence length>, 4096])]
    #     Batch size is always 1.
    #     """
    #     pass


# https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
class LLaMA2_13B_ChatHF(LLM):
    def __init__(self, sampling_seed: int, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("meta-llama/Llama-2-13b-chat-hf", sampling_seed, quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = LlamaForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.name, **self.tokenizer_config)

    def extend_generation_config(self, generation_config: dict) -> None:
        pass

    # def extract_internal_states_from_output(self, output) -> Dict[str, Any]:
    #     """
    #     Shapes:
    #         - output.logits: [torch.Size([1, <sequence length>, 32000])]
    #         - output.hidden_states: [41, torch.Size([1,  <sequence length>, 5120])]
    #     Batch size is always 1.
    #     """
    #     pass
