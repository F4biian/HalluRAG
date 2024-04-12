import torch
from typing import Tuple

class LLM:
    def __init__(self, default_temperature: float=0.0, auto_load: bool=False) -> None:
        self.default_temperature = default_temperature

        self.loaded = False
        self.model = None
        self.tokenizer = None

        if auto_load:
            self.load()

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

    def get_internal_states(self, prompt: str, llm_output: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert text to tokens
        model_inputs = self.to_model_inputs(prompt, llm_output=llm_output)

        # Retrieve internal states
        output = self.model(model_inputs.to(self.model.device), output_hidden_states=True, output_attentions=True)
        print(type(output))

        return output.logits, output.hidden_states, output.attentions
    
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