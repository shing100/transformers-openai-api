# transformers-openai-api

`transformers-openai-api`는 'https://github.com/jquesnelle/transformers-openai-api'을 참고참고하였습니다. 로컬에서 실행되는 NLP [transformers](https://github.com/huggingface/transformers/) 모델을 [OpenAI Completions API](https://beta.openai.com/docs/api-reference/completions) 호환 서버로 호스팅하는 도구입니다. 이를 통해 `transformers` 모델을 실행하고 [OpenAI Python Client](https://github.com/openai/openai-python)나 [LangChain](https://github.com/hwchase17/langchain)과 같은 OpenAI 도구와 호환되는 API를 통해 제공할 수 있습니다.

## 특징

- OpenAI API 호환 엔드포인트 (`/v1/completions`, `/v1/chat/completions`)
- 다양한 `transformers` 모델 지원 (Seq2Seq, CausalLM)
- 사용자 정의 가능한 채팅 템플릿
- 간편한 설정 및 배포

## 설치

### 소스에서 설치

```sh
git clone https://github.com/yourusername/transformers-openai-api
cd transformers-openai-api
pip install -e .
```

## 빠른 시작

1. 설정 파일 생성:

   `config.json` 파일을 생성하고 다음과 같이 설정합니다:

   ```json
   {
     "MODELS": {
       "gpt-3.5-turbo": {
         "NAME": "CarrotAI/Carrot-Ko-2B-Instruct",
         "TYPE": "CausalLM",
         "MODEL_CONFIG": {
           "device_map": "auto",
           "trust_remote_code": true,
           "torch_dtype": "float16"
         },
         "CHAT_TEMPLATE": "vicuna"
       }
     },
     "HOST": "0.0.0.0",
     "PORT": 13000
   }
   ```

2. 서버 실행:

   ```sh
   transformers-openai-api --config /path/to/your/config.json
   ```

   설정 파일을 지정하지 않으면 패키지에 포함된 기본 설정 파일을 사용합니다.

## OpenAI Python Client와 함께 사용하기

`OPENAI_API_BASE` 환경 변수를 `http://HOST:PORT/v1`로 설정하거나 `openai` 객체의 `api_base` 속성을 직접 설정합니다:

```python
import openai
openai.api_base = 'http://localhost:13000/v1'
```

## 설정

모든 설정은 `config.json`을 통해 관리됩니다. 주요 설정 항목은 다음과 같습니다:

- `MODELS`: 사용할 모델 설정
- `HOST`: 서버 호스트 (기본값: 127.0.0.1)
- `PORT`: 서버 포트 (기본값: 5000)
- `BEARER_TOKENS`: API 인증을 위한 토큰 목록 (선택사항)

## 채팅 템플릿

채팅 형식은 Jinja2 템플릿을 사용하여 구성할 수 있습니다. 템플릿 파일은 `transformers_openai_api/chat_templates/` 디렉토리에 위치합니다. 설정 파일에서 `CHAT_TEMPLATE` 필드를 사용하여 원하는 템플릿을 지정할 수 있습니다.

## 개발

1. 저장소 클론:
   ```
   git clone https://github.com/yourusername/transformers-openai-api.git
   cd transformers-openai-api
   ```

2. 가상 환경 생성 및 활성화:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. 의존성 설치:
   ```
   pip install -e .
   ```

4. 개발 모드로 실행:
   ```
   transformers-openai-api --config /path/to/your/config.json
   or
   CUDA_VISIBLE_DEVICES='0,1' transformers-openai-api --config /path/to/your/config.json
   ```

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다. 중요한 변경사항에 대해서는 먼저 이슈를 열어 논의해 주세요.

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.