from models.llm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# For activation value extraction do the following:
# Go to the transformers package on your local disk (e.g. ~/.local/lib/python3.10/site-packages/transformers/models/gemma/modeling_gemma.py)
# In line 189 (version: transformers @ git+https://github.com/huggingface/transformers.git@8127f39624f587bdb04d55ab655df1753de7720a)
# replace the inner block of the forward method with this:
'''
# Original version
# return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# Version taken from MIND (https://github.com/oneal2000/MIND/issues/2):
a = self.act_fn(self.gate_proj(x))
self.activation_values_from_inserted_code = a.clone().detach()
return self.down_proj(a * self.up_proj(x))
'''

# https://huggingface.co/google/gemma-7b
class Gemma_7B(LLM):
    def __init__(self, sampling_seed: int, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("google/gemma-7b", sampling_seed, quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_config)
    
    def extend_generation_config(self, generation_config: dict) -> None:
        pass

    # def extract_internal_states_from_output(self, output) -> Dict[str, Any]:
    #     """
    #     Shapes:
    #         - output.logits: [torch.Size([1, <sequence length>, 256000])]
    #         - output.hidden_states: [29, torch.Size([1,  <sequence length>, 3072])]
    #     Batch size is always 1.
    #     """
    #     pass

# https://huggingface.co/google/gemma-7b-it
class Gemma_7B_Instruct(LLM):
    def __init__(self, sampling_seed: int, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("google/gemma-7b-it", sampling_seed, quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **self.tokenizer_config)
    
    def extend_generation_config(self, generation_config: dict) -> None:
        pass

    # def extract_internal_states_from_output(self, output) -> Dict[str, Any]:
    #     """
    #     Shapes:
    #         - output.logits: [torch.Size([1, <sequence length>, 256000])]
    #         - output.hidden_states: [29, torch.Size([1,  <sequence length>, 3072])]
    #     Batch size is always 1.
    #     """
    #     pass