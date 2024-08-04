import json
import re
import time
import uuid
import torch
from typing import Any, Callable, Mapping, Optional
from flask import Flask, make_response, request, abort, jsonify
from functools import wraps
from .models import CausalLM, Model, Seq2Seq
from .metrics import Metrics
from .utils import apply_chat_template, load_chat_template


app = Flask(__name__)
models = {}
metrics: Optional[Metrics]

def extract_assistant_response(text):
    match = re.search(r'assistant\s*\n([\s\S]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Assistant's response not found."


def check_token(f: Callable):
    @wraps(f)
    def decorator(*args, **kwargs):
        bearer_tokens = app.config.get('BEARER_TOKENS')
        if bearer_tokens is None:
            return f(*args, **kwargs)

        authorization = request.headers.get('Authorization')
        if authorization and authorization.startswith('Bearer '):
            token = authorization[7:]
            if token in bearer_tokens:
                return f(*args, **kwargs)
        return make_response(jsonify({
            'message': 'Invalid token'
        }), 401)

    return decorator


@app.route('/v1/chat/completions', methods=['POST'])
@check_token
def chat_completion():
    data = request.json
    model_name = data.get('model')
    messages = data.get('messages', [])

    if not model_name or not messages:
        return jsonify({"error": "model and messages are required"}), 400

    model: Model = models.get(model_name)
    if not model:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    if not isinstance(model, CausalLM):
        return jsonify({"error": f"Model {model_name} does not support chat completions"}), 400

    # Apply chat template
    prompt = apply_chat_template(model.chat_template, messages)

    # Generate response
    response = model.generate(prompt)

    # Extract only the assistant's response
    assistant_response = extract_assistant_response(response['text'])

    result = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_response,
            },
            "finish_reason": "stop"
        }],
        "usage": response['usage']
    }

    # Update metrics if enabled
    global metrics
    if metrics is not None:
        metrics.update(result)

    return jsonify(result)


def convert_model_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    config = {}
    if val is not None:
        for key, value in val.items():
            if key == 'torch_dtype':
                if value == 'float16':
                    config['torch_dtype'] = torch.float16
                elif value == 'float32':
                    config['torch_dtype'] = torch.float32
                elif value == 'int8':
                    config['torch_dtype'] = torch.int8
                else:
                    raise RuntimeError(
                        f"Unknown torch_dtype {config['torch_dtype']}")
            else:
                config[key] = value
    return config


def convert_tokenizer_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return val if val is not None else {}


def convert_generate_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    config = {}
    if val is not None:
        for key, value in val.items():
            if key == 'max_tokens':
                config['max_length'] = value
            else:
                config[key] = value
    return config


def convert_decode_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return val if val is not None else {}


def completion(model_name: str):
    global id
    this_id = id
    id += 1

    model: Model = models[model_name]

    response = model.completions(convert_generate_config(request.json))
    response.update({
        'object': 'text_completion',
        'model': model_name,
        'created': int(time.time()),
        'id': f'cmpl-{this_id}'
    })

    global metrics
    if metrics is not None:
        metrics.update(response)

    return make_response(jsonify(response))


@app.route('/v1/engines')
def v1_engines():
    return make_response(jsonify({
        'data': [{
            'object': 'engine',
            'id': id,
            'ready': True,
            'owner': 'openai',
            'permissions': None,
            'created': None
        } for id in models.keys()]
    }))


@app.route('/v1/completions', methods=['POST'])
@check_token
def v1_completions():
    return completion(request.json['model'])


@app.route('/v1/engines/<model_name>/completions', methods=['POST'])
@check_token
def engine_completion(model_name: str):
    return completion(model_name)


@app.route('/v1/metrics')
def metrics_():
    global metrics
    if metrics is None:
        abort(404)

    return make_response(jsonify(metrics.get()))


def make_transformers_openai_api(config_path: str) -> Flask:
    app.config.from_file(config_path, load=json.load)

    if app.config.get('METRICS', 1) != 0:
        global metrics
        metrics = Metrics()

    for mapping, config in app.config['MODELS'].items():
        if config.get('ENABLED', True) == False:
            continue
        model_config = convert_model_config(config.get('MODEL_CONFIG'))
        model_device = config.get('MODEL_DEVICE', 'cuda')
        tokenizer_config = convert_tokenizer_config(
            config.get('TOKENIZER_CONFIG'))
        tokenizer_device = config.get('TOKENIZER_DEVICE', 'cuda')
        generate_config = convert_generate_config(
            config.get('GENERATE_CONFIG'))
        decode_config = convert_decode_config(
            config.get('DECODE_CONFIG'))
        if config['TYPE'] == 'Seq2Seq':
            models[mapping] = Seq2Seq(
                config['NAME'], model_config, model_device, tokenizer_config, tokenizer_device, generate_config, decode_config)
        elif config['TYPE'] == 'CausalLM':
            chat_template_name = config.get('CHAT_TEMPLATE')
            if chat_template_name:
                chat_template = load_chat_template(chat_template_name)
            else:
                chat_template = ''
            models[mapping] = CausalLM(
                config['NAME'], model_config, model_device, tokenizer_config, tokenizer_device, generate_config, decode_config, chat_template)
        else:
            raise RuntimeError(f'Unknown model type {config["TYPE"]}')

    return app