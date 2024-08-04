from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from .utils import apply_chat_template
import torch
import logging

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
        self.generate_config = generate_config
        self.decode_config = decode_config
        self.tokenizer_device = tokenizer_device
        self.chat_template = chat_template
        self.model_device = model_device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
        logger.info(f"Model pad_token_id: {self.model.config.pad_token_id}")

    @torch.no_grad()
    def generate(self, input_text: str) -> Mapping[str, Any]:
        tokenized_input = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized_input.input_ids.to(self.model_device)
        attention_mask = tokenized_input.attention_mask.to(self.model_device)

        # Generate output
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.generate_config
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

    def chat_completions(self, messages: List[Mapping[str, str]]) -> Mapping[str, Any]:
        prompt = apply_chat_template(self.chat_template, messages)
        return self.generate(prompt)