import requests
import json
import re

BASE_URL = "http://localhost:13000/v1"


def extract_assistant_response(text):
    # 'assistant' 다음에 오는 모든 내용을 추출합니다.
    match = re.search(r'assistant\s*\n([\s\S]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Assistant's response not found."


def test_chat_completions():
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "CarrotAI/Carrot-Ko-2B-Instruct",  # config.json에 설정된 모델 이름으로 변경하세요
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello?"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Chat Completions Test: SUCCESS")
        full_content = response.json()['choices'][0]['message']['content']
        assistant_response = extract_assistant_response(full_content)
        print("Assistant's response:")
        print(assistant_response)
    else:
        print("Chat Completions Test: FAILED")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


if __name__ == "__main__":
    print("Testing transformers-openai-api")
    print("================================")

    test_chat_completions()