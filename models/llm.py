import torch
from typing import Tuple
from transformers import QuantoConfig

from models.utils import get_shape

class LLM:
    hf_token: str=None

    def __init__(self, name: str, quantization: str=None, default_temperature: float=0.0, auto_load: bool=False) -> None:
        """
        quantization: one of "float8", "int8", "int4", "int2" TODO: rewrite this doc string with gpt
        """
        self.name = name
        self.quantization = quantization
        self.default_temperature = default_temperature

        self.loaded = False
        self.model = None
        self.tokenizer = None
        
        self.model_config = {
            "device_map": "auto"
        }
        self.tokenizer_config = {
            "device_map": "auto"
        }

        if LLM.hf_token:
            self.model_config["token"] = LLM.hf_token
            self.tokenizer_config["token"] = LLM.hf_token

        if quantization:
            self.model_config["quantization_config"] = QuantoConfig(weights=quantization)

        if auto_load:
            self.load()

    def __enter__(self) -> None:
        self.load()
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.unload()

    def __str__(self) -> str:
        if self.quantization:
            return f"{self.name} ({self.quantization})"
        else:
            return self.name

    def load(self) -> None:
        if not self.loaded:
            self.loaded = True
            self._load()

    def unload(self) -> None:
        if self.loaded:
            self.loaded = False
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

    def tokenize(self, input_str: str) -> torch.Tensor:
        return self.tokenizer([input_str], return_tensors="pt").to(self.model.device)

    def detokenize(self, ids: torch.Tensor, skip_special_tokens: bool=True) -> str:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)[0]

    def tokenize_chat(self, chat):
        encodeds = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
        return encodeds.to(self.model.device)

    def generate(self, prompt: str, max_new_tokens=1000, temperature=None, do_sample=False) -> str:
        # If temperature not set, then use default temperature
        temperature = temperature if temperature else self.default_temperature

        # Settings for generating the output
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature
        }
        self.extend_generation_config(generation_config)

        # Convert text to tokens
        model_inputs = self.to_model_inputs(prompt)

        # Generate output
        generated_ids = self.model.generate(model_inputs, **generation_config)

        # Remove prompt tokens from output
        generated_ids = generated_ids[:, len(model_inputs[0]):]

        # Return generated ids as text without special tokens (e.g. eos or bos)
        return self.detokenize(generated_ids, skip_special_tokens=True)

    def get_internal_states(self, prompt: str, llm_output: str, system: str=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert text to tokens
        model_inputs = self.to_model_inputs(prompt, llm_output=llm_output, system=system)

        print(model_inputs.size())

        # Retrieve internal states
        output = self.model(model_inputs.to(self.model.device), output_hidden_states=True, output_attentions=False)

        # This even works if a model does not fit into the GPU, but only returns logits for last token. That is why the upper method is employed.
        # output = self.model.generate(input_ids=model_inputs, max_new_tokens=1, output_logits=True, output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)

        """
        Example sizes of LLaMA 7B:
            logits:          [torch.Size([1, 37, 32000])]
            hidden_states:   [33, torch.Size([1, 37, 4096])]
        """

        print(get_shape(output.logits))
        print(get_shape(output.hidden_states))

        # TODO: put this into every subclass and always return something related to 4096 (matmult for smaller or bigger ones to get to 4096)

        hidden_states = []
        for hidden_state in output.hidden_states:
            hidden_states.append(hidden_state.clone().detach().tolist())

        # always the last token, the rest is not important?
        # calc pp and pe immediately here without returning logits
        # calc mean over all 32 layers, so at the only a 4096  vector
        # and seperately each layer? or just every 8th or just the last one?

        return output.logits.clone().detach().tolist(), hidden_states
    
    def to_model_inputs(self, prompt: str, system: str=None, llm_output: str=None) -> torch.Tensor:
        chat = []

        # Add system message to chat
        if system:
            chat.append({
                "role": "system",
                "content": system
            })

        # Add user's prompt to chat
        chat.append({
            "role": "user",
            "content": prompt
        })

        # Add output of LLM to chat
        if llm_output:
            chat.append({
                "role": "assistant",
                "content": llm_output
            })
            
        return self.tokenize_chat(chat)
    
    def _load(self) -> None:...
    
    def extend_generation_config(self, generation_config: dict) -> None:...