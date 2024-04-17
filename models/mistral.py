from models.llm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
class Mistral_7B_Instruct_V1(LLM):
    def __init__(self, sampling_seed: int, quantization: str = None, default_temperature: float = 0, auto_load: bool = False) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", sampling_seed, quantization, default_temperature, auto_load)

    def _load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side="left", **self.tokenizer_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extend_generation_config(self, generation_config: dict) -> None:
        generation_config["pad_token_id"] = self.tokenizer.eos_token_id

    # def extract_internal_states_from_output(self, output) -> Dict[str, Any]:
    #     """
    #     Shapes:
    #         - output.logits: [torch.Size([1, <sequence length>, 32000])]
    #         - output.hidden_states: [33, torch.Size([1,  <sequence length>, 4096])]
    #     Batch size is always 1.
    #     """
    #     pass