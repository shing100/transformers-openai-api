import requests
import json
import os

# 서버 URL 설정
BASE_URL = "http://localhost:13000/v1"  # 포트는 config.json에 설정된 값으로 변경하세요


def test_chat_completions():
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "Carrot-Ko-2B-Instruct",  # config.json에 설정된 모델 이름으로 변경하세요
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like today?"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Chat Completions Test: SUCCESS")
        print("Response:", json.dumps(response.json(), indent=2))
    else:
        print("Chat Completions Test: FAILED")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


def test_models():
    url = f"{BASE_URL}/models"
    headers = {}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("Models Test: SUCCESS")
        print("Available Models:", json.dumps(response.json(), indent=2))
    else:
        print("Models Test: FAILED")
        print("Status Code:", response.status_code)
        print("Response:", response.text)


if __name__ == "__main__":
    print("Testing transformers-openai-api")
    print("================================")

    test_models()
    print("\n")
    test_completions()
    print("\n")
    test_chat_completions()