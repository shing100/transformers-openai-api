import os
from jinja2 import Template


def load_chat_template(template_name: str) -> str:
    template_dir = os.path.join(os.path.dirname(__file__), 'chat_templates')
    template_path = os.path.join(template_dir, f"{template_name}.jinja")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Chat template file not found: {template_path}")

    with open(template_path, 'r') as file:
        return file.read()


def apply_chat_template(template: str, messages: list, add_generation_prompt: bool = True, **kwargs) -> str:
    jinja_template = Template(template)
    return jinja_template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)