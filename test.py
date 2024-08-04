import requests
import json

BASE_URL = "http://localhost:13000/v1"
API_KEY = "your-api-key"  # 필요한 경우 실제 API 키로 변경하세요


def test_chat_completions():
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"  # BEARER_TOKENS를 사용하지 않는 경우 이 줄을 제거하세요
    }
    data = {
        "model": "Carrot-Ko-2B-Instruct",  # config.json에 설정된 모델 이름으로 변경하세요
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "안녕?"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Chat Completions Test: SUCCESS")
        assistant_response = response.json()['choices'][0]['message']['content']
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