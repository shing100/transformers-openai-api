from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .utils import apply_chat_template
import torch
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prompts(request: Mapping[str, Any]) -> List[str]:
    prompt = request['prompt']
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def _completions_auto(
        request: Mapping[str, Any],
        tokenizer: Any,
        tokenizer_device: Optional[str],
        model: Any,
        generate_config: Mapping[str, Any],
        decode_config: Mapping[str, Any],
        auto_echo: bool):
    generate_args = {}
    generate_args.update(generate_config)
    generate_args.update(request)

    decode_args = {
        "skip_special_tokens": True
    }
    decode_args.update(decode_config)

    if ('top_p' in generate_args or 'top_k' in generate_args or 'temperature' in generate_args) and 'do_sample' not in generate_args:
        generate_args['do_sample'] = True
        if generate_args.get('temperature', 1.0) == 0:
            generate_args.pop('temperature', None)
        elif generate_args.get('top_p', 1.0) == 1.0:
            generate_args.pop('top_p', None)
        if 'top_k' not in generate_args:
            generate_args['top_k'] = 0

    prompts = get_prompts(generate_args)
    echo = generate_args.get('echo', False)
    n = generate_args.get('n', 1)

    generate_args.pop('model', None)
    generate_args.pop('prompt', None)
    generate_args.pop('n', None)

    # TODO
    generate_args.pop('best_of', None)
    generate_args.pop('presence_penalty', None)
    generate_args.pop('frequency_penalty', None)
    generate_args.pop('logit_bias', None)

    inputs = []
    prompt_tokens_count = 0
    for prompt in prompts:
        input = tokenizer(prompt, return_tensors="pt").input_ids
        if tokenizer_device is not None:
            input = input.to(tokenizer_device)
        prompt_tokens_count += input.size(dim=1)
        inputs.append(input)

    choices = []
    completion_tokens_count = 0
    for i in range(0, len(inputs)):
        for _ in range(0, n):
            output = model.generate(inputs[i], **generate_args)[0]
            completion_tokens_count += len(output)
            text = tokenizer.decode(output, **decode_args)
            if echo and not auto_echo:
                text = prompts[i] + text
            choices.append({
                'text': text,
                'index': i,
            })

    return {
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens_count,
            'completion_tokens': completion_tokens_count,
            'total_tokens': prompt_tokens_count + completion_tokens_count
        }
    }


class Model(ABC):

    @abstractmethod
    def chat_completions(self, messages: List[Mapping[str, str]]) -> Mapping[str, Any]:
        pass

class Seq2Seq(Model):
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    generate_config: Mapping[str, Any]
    decode_config: Mapping[str, Any]
    tokenizer_device: Optional[str]

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_config: Mapping[str, Any],
            model_device: Optional[str],
            tokenizer_config: Mapping[str, Any],
            tokenizer_device: Optional[str],
            generate_config: Mapping[str, Any],
            decode_config: Mapping[str, Any]) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path, **model_config)
        if model_device is not None:
            self.model = self.model.to(model_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        self.generate_config = generate_config
        self.decode_config = decode_config
        self.tokenizer_device = tokenizer_device

    def completions(self, request) -> List[str]:
        return _completions_auto(request, self.tokenizer, self.tokenizer_device, self.model, self.generate_config, self.decode_config, False)


def load_model_config(model_name: str) -> dict:
    config_dir = os.path.join(os.path.dirname(__file__), 'generation_configs')
    config_path = os.path.join(config_dir, f"{model_name}.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


class CausalLM(Model):
    def __init__(self, pretrained_model_name_or_path: str,
                 model_config: Mapping[str, Any],
                 model_device: Optional[str],
                 tokenizer_config: Mapping[str, Any],
                 tokenizer_device: Optional[str],
                 generate_config: Mapping[str, Any],
                 decode_config: Mapping[str, Any],
                 chat_template: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_config)
        if model_device is not None:
            self.model = self.model.to(model_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model-specific config
        model_name = os.path.basename(pretrained_model_name_or_path)
        model_specific_config = load_model_config(model_name)

        # Prepare generation config
        generation_config_dict = {
            "chat_template": model_specific_config.get("chat_template", chat_template),
            "stop_token_ids": model_specific_config.get("stop_token_ids", []),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_config_dict.update(generate_config)
        self.generation_config = GenerationConfig(**generation_config_dict)

        self.decode_config = decode_config
        self.tokenizer_device = tokenizer_device
        self.chat_template = self.generation_config.chat_template
        self.stop_token_ids = self.generation_config.stop_token_ids

    @torch.no_grad()
    def generate(self, input_text: str) -> Mapping[str, Any]:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.tokenizer_device)
        attention_mask = inputs.attention_mask.to(self.tokenizer_device)

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            stopping_criteria=self.get_stopping_criteria()
        )

        response = self.tokenizer.decode(output[0], **self.decode_config)

        return {
            "text": response,
            "usage": {
                "prompt_tokens": input_ids.size(1),
                "completion_tokens": output.size(1) - input_ids.size(1),
                "total_tokens": output.size(1)
            }
        }

    def get_stopping_criteria(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_token_ids):
                self.stop_token_ids = stop_token_ids

            def __call__(self, input_ids, scores, **kwargs):
                for stop_id in self.stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        return StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])

    def chat_completions(self, messages: List[Mapping[str, str]]) -> Mapping[str, Any]:
        prompt = apply_chat_template(self.chat_template, messages)
        return self.generate(prompt)