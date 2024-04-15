import torch
from typing import Any, Dict, Tuple, List
from transformers import QuantoConfig

from models.utils import get_pe, get_shape

class LLM:
    """
    Large Language Model (LLM) class for loading and initializing transformer models.

    Args:
        name (str): The name of the model.
        quantization (str, optional): One of "float8", "int8", "int4", "int2".
            Defaults to None.
        default_temperature (float, optional): The default temperature for sampling text.
            Defaults to 0.0.
        auto_load (bool, optional): Whether to automatically load the model.
            Defaults to False.
    
    Attributes:
        name (str): The name of the model.
        quantization (str): The quantization method used for the model.
        default_temperature (float): The default temperature for sampling text.
        loaded (bool): Indicates whether the model is loaded.
        model (object): The loaded model object.
        tokenizer (object): The tokenizer object associated with the model.
        model_config (dict): Configuration parameters for the model.
        tokenizer_config (dict): Configuration parameters for the tokenizer.
        hf_token (str): Hugging Face API token for model access. This attribute is shared among all instances of the class.
    """

    hf_token: str = None

    def __init__(
        self,
        name: str,
        quantization: str = None,
        default_temperature: float = 0.0,
        auto_load: bool = False,
    ) -> None:
        """
        Initialize LLM with the provided parameters.
        """
        self.name = name
        self.quantization = quantization
        self.default_temperature = default_temperature

        self.loaded = False
        self.model = None
        self.tokenizer = None
        
        self.model_config = {
            "device_map": 'auto'
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
        """
        Try to load the model when used in a "with" statement.
        """
        self.load()
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """
        Try to close the model once a "with" statement is finished or has been abruptly terminated.
        """
        self.unload()

    def __str__(self) -> str:
        """
        Return a readable string representation for the instance of this class.
        """
        if self.quantization:
            return f"{self.name} ({self.quantization})"
        else:
            return self.name

    def load(self) -> None:
        """
        Load the model into memory. Once loaded, this function does not do anything, preventing the model from being loaded multiple times.
        """
        if not self.loaded:
            self.loaded = True
            self._load()

    def unload(self) -> None:
        """
        Remove the model from memory. Once removed, this function does not do anything, preventing the model from being unloaded multiple times.
        """
        if self.loaded:
            self.loaded = False
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

    def tokenize(self, input_str: str) -> torch.Tensor:
        """
        Convert a string to its token ids.
        """
        return self.tokenizer([input_str], return_tensors="pt").to(self.model.device)

    def detokenize(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Convert token ids to the corresponding string.
        """
        return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)[0]

    def tokenize_chat(self, chat):
        """
        Tokenize a conversation chat.
        """
        encodeds = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
        return encodeds

    def generate(self, prompt: str, max_new_tokens=1000, temperature=None, do_sample=False) -> str:
        """
        Generate text based on the provided prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
            temperature (float, optional): The temperature parameter for sampling. Defaults to None.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.

        Returns:
            str: The generated text of the LLM.
        """
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

    def get_internal_states(self, prompt: str, llm_output: str, system: str = None) -> Dict[str, Any]:
        """
        Retrieve the internal states of the model.

        Args:
            prompt (str): The input prompt for generating text.
            llm_output (str): The output generated by the model.
            system (str, optional): System message. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the internal states of the model.
        """

        # Convert text to tokens
        model_inputs = self.to_model_inputs(prompt, llm_output=llm_output, system=system)
        input_ids = model_inputs[0].tolist()

        # Convert text (without LLM output) to token in order to find the index of the very first token the LLM has generated 
        prompt_inputs = self.to_model_inputs(prompt, system=system)
        llm_output_starts_at = prompt_inputs.size()[1]

        # Retrieve internal states
        output = self.model(torch.tensor(model_inputs.tolist()).to(self.model.device), output_hidden_states=True)

        return self.extract_internal_states_from_output(output, input_ids, llm_output_starts_at)
    
    def to_model_inputs(self, prompt: str, system: str = None, llm_output: str = None) -> torch.Tensor:
        """
        Convert the prompt, system message, and LLM output into model inputs.

        Args:
            prompt (str): User input prompt.
            system (str, optional): System message. Defaults to None.
            llm_output (str, optional): LLM output. Defaults to None.

        Returns:
            torch.Tensor: Model inputs as tokenized tensors.
        """
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
    
    def _load(self) -> None:
        """
        Load the model into memory.
        """
        ...

    def extend_generation_config(self, generation_config: dict) -> None:
        """
        Extend the generation configuration with LLM specific settings.

        Args:
            generation_config (dict): Generation configuration dictionary.
        """
        ...

    def _percentile_layer_last_token(self, layers: Tuple[torch.Tensor], percentile: float) -> List[float]:
        max_layer_index = len(layers)-1
        return layers[int(max_layer_index * percentile)][0][-1].clone().detach().tolist()
    
    def _percentile_layer_mean_last_token(self, layers: Tuple[torch.Tensor], from_percentile: float, to_percentile: float) -> List[float]:
        max_layer_index = len(layers)-1
        start_i = int(max_layer_index * from_percentile)
        end_i   = int(max_layer_index * to_percentile)

        layer_sum = layers[start_i][0][-1].clone().detach()
        for i in range(start_i+1, end_i+1):
            layer_sum += layers[i][0][-1].clone().detach()
        layer_sum = layer_sum / (end_i - start_i)

        return layer_sum.tolist()

    def extract_internal_states_from_output(self, output, input_ids: List[int], llm_output_starts_at: int) -> Dict[str, Any]:
        """
        Extract internal states from model output.

        Args:
            output: Model output.
            input_ids (list): Token ids of the input that has been given to the model.
            llm_output_starts_at (int): Index of the first token the LLM has generated.

        Returns:
            Dict[str, Any]: Dictionary containing internal states.
        """

        layers = output.hidden_states[1:]

        # print(get_shape(output.logits))
        # print(get_shape(output.hidden_states))

        probability, entropy = get_pe(output.logits, input_ids, llm_output_starts_at)
        
        states = {
            "layer_0_last_token": self._percentile_layer_last_token(layers, 0.0),
            "layer_25_last_token": self._percentile_layer_last_token(layers, 0.25),
            "layer_50_last_token": self._percentile_layer_last_token(layers, 0.5),
            "layer_75_last_token": self._percentile_layer_last_token(layers, 0.75),
            "layer_100_last_token":self._percentile_layer_last_token(layers, 1.0),
            "layers_mean_25_last_token": self._percentile_layer_mean_last_token(layers, 0.0, 0.25),
            "layers_mean_50_last_token": self._percentile_layer_mean_last_token(layers, 0.25, 0.5),
            "layers_mean_75_last_token": self._percentile_layer_mean_last_token(layers, 0.5, 0.75),
            "layers_mean_100_last_token":self._percentile_layer_mean_last_token(layers, 0.75, 1.0),
            "probability": probability,
            "entropy": entropy
        }

        return states
    
