import os

import aiohttp
import asyncio
import json
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

GPT_API_KEY = os.getenv("OPENAI_API_KEY")

# 입력 JSONL 파일과 출력 JSONL 파일 경로 설정
input_file = "hsu_mdai_training_data_answer_reduced_18.jsonl"
output_file = "hsu_mdai_training_data_answer_reduced_pair_18.jsonl"

MODEL = "gpt-3.5-turbo"  # 사용할 GPT 모델 지정 "gpt-4o-mini" "gpt-3.5-turbo"

# 질문 데이터, 혹은 답변 데이터 중 어떤 것을 다룰 건지 선택
QUESTION_PROCESS = False

PROMPT_FOR_ANSWER_MAKER = \
    "You are an emergency medical expert who has received contact from a patient or their guardian \
    in a critical situation. Please provide brief and essential medical advice based on reliable information. \
    , and avoid unnecessary lengthy explanations due to the urgency of the situation."

PROMPT_FOR_QUIZ_MAKER = \
    "You are an emergency medical expert. Based on the responses provided by another emergency medical expert \
    to a patient’s question, use your expertise to infer what the patient or their guardian might have asked \
    during the critical situation. Considering the urgency of the situation, focus on key details \
    when inferring the likely question. The inferred question must always be in Korean \
    and must contain only the question itself without any additional phrases or formatting."


if QUESTION_PROCESS:
    PROMPT_FOR_LLM = PROMPT_FOR_ANSWER_MAKER
    generate_target = "completion"
    generate_method = "prompt"
else:
    PROMPT_FOR_LLM = PROMPT_FOR_QUIZ_MAKER
    generate_target = "prompt"
    generate_method = "completion"


# 비동기적으로 GPT API 호출하여 답변 생성하는 함수
# async와 await은 비동기 프로그래밍을 구현하기 위해 사용하는 문법
# 비동기 프로그래밍: 작업을 시작한 후 결과를 기다리지 않고 다른 작업을 진행, 시간 절약 가능
# async def로 비동기적으로 실행될 수 있는 함수를 정의
# async 함수는 호출 시 실행 결과 즉시 반환 않고 코루틴 객체 반환
# 코루틴 객체가 실제로 실행되기 위해서는 await 키워드나 asyncio.run 메서드 필요
# await 키워드는 async 함수 내부에서만 사용 가능한 키워드
# 비동기 작업(코루틴)을 실행 후 작업이 완료될 때까지 기다림
async def generate_response(session, data, model=MODEL):
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",  # OpenAI API 접근 위한 인증 토큰
        "Content-Type": "application/json",  # API가 받는 데이터의 형식 표기(사실 1개 뿐이라 고정)
    }
    payload = {
        "model": model,  # 사용할 GPT 모델 지정 "gpt-4o-mini" "gpt-3.5-turbo"
        "messages": [
            {"role": "system", "content": PROMPT_FOR_LLM},
            {"role": "user", "content": data[generate_method]}  # GPT 모델이 응답 생성 시 사용할 프롬프트(모델에게 질문 또는 맥락 정보를 제공)
        ],
        "max_tokens": 300,  # 생성될 응답의 최대 길이 설정(단위 토큰, 한국어 1000자, 1문단 정도)
        "temperature": 0.7,  # 모델 응답의 창의성 조정하는 파라메타, 챗봇이라 조금 높게 설정
        "n": 1,  # 생성할 응답의 개수는 1개
        # "stop": ["###"]  # 모델 응답을 종료할 문자열, JSONL의 prompt에서 끝에 '###' 사용했음
    }
    try:
        # async with session.post()는 비동기 HTTP POST 요청을 보내는 함수
        # OpenAI API 같은 HTTP 서버에 데이터 전송하고 응답을 받을 때 사용
        # async with 문이 네트워크 연결 종료나 예외 발생 시에도 리소스 누수 발생 않도록 보장
        # session.post는 HTTP POST 요청을 보내는 역할
        # session은 aiohttp.ClientSession 객체인 HTTP 세션
        # .post는 데이터를 서버에 전송할 때 사용하는 HTTP 요청 메서드
        # .post 1번 매개변수: 요청을 보낼 대상 URL
        # 2번 매개변수 headers: HTTP 요청에 포함되는 헤더 정보(인증 토큰, 데이터 형식)
        # 3번 매개변수 json: HTTP 요청의 본문인 데이터
        # .post의 반환값은 HTTP 응답 객체 -> response
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers,
                                json=payload) as response:
            if response.status == 200:  # response.status는 openai 서버에서 받아온 응답 객체의 상태 코드
                # response.status=200이면 성공, 400이면 필수 항목 누락, 401이면 API 키 오류, 429면 요청 제한 초과, 500이면 서버 오류
                result = await response.json()  # 응답 데이터를 .json()이용 JSON 형식으로 변환, await로 비동기적으로 읽음
                # openai api 통해 생성된 응답을 data의 completion에 저장
                # 생성된 응답에서 text의 첫번째 응답 가져오고 양쪽 공백 제거
                # data["completion"] = result["choices"][0]["text"].strip()
                data[generate_target] = result["choices"][0]["message"]["content"].strip()
                return data  # 생성된 응답 넣어 처리 완료된 data 반환
            else:  # 정상적이지 못한 응답 받음, 에러 메시지 출력
                print(f"Error: {response.status} - {await response.text()}")  # HTTP 상태 코드와 API 서버가 반환한 에러 메시지 출력
                return None
    except Exception as e:  # 예외 처리 코드
        print(f"Exception during API call: {e}")  # 예외 메시지 출력
        return None


# JSONL 파일을 읽고 비동기로 GPT API 호출 결과를 처리하는 함수
# 매개변수 batch_size는 한 번에 비동기로 처리할 요청 수
async def process_file(input_path, output_path, batch_size):
    # JSONL 파일을 읽고 각 줄을(줄 하나가 JSON 파일) 리스트 lines로 저장
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()  # JSONL 파일 내용을 줄 단위로 읽어 리스트로 반환

    tasks = []  # 비동기 작업(코루틴)들 저장하는 리스트, batch_size 이상 쌓이면 처리
    results = []  # 처리된 결과 저장하는 리스트

    # OpenAI API 호출을 위한 비동기 HTTP 연결 관리할 세션 객체 생성
    async with aiohttp.ClientSession() as session:  # aiohttp.ClientSession은 HTTP 요청을 효율적으로 처리하는 객체(재사용 가능)
        for line in tqdm(lines, desc="Processing JSONL"):
            data = json.loads(line.strip())  # JSONL의 각 줄을 JSON으로 변환

            # 요청이 필요한 경우만 처리
            if not data.get(generate_target):  # completion이 없는 데이터만 처리 대상(파이썬에서는 빈 문자열 ""도 False)
                tasks.append(generate_response(session, data, MODEL))  # GPT API 요청하여 코루틴 객체로 만들고 task 리스트에 추가

            # batch_size보다 많이 쌓이면 배치 처리
            # asyncio.gather()은 코루틴들을 동시에 실행 후 완료된 결과를 반환(병렬 처리 시행하는 코드)
            # *은 언패킹 연산자, iterable한 객체의 모든 요소를 개별적인 인수로 전달하기 위해 사용(리스트 직접 전달 시 리스트 하나를 인수로 취급)
            if len(tasks) >= batch_size:
                results += await asyncio.gather(*tasks)
                tasks = []  # tasks 빈 리스트로 초기화(리스트 비우기)

        # 루프 종료 후, 남은 코루틴 있으면 batch_size보다 작아도 한꺼번에 처리
        if tasks:
            results += await asyncio.gather(*tasks)

    # 처리된 results 리스트를 JSONL 형식으로 출력 파일에 저장
    with open(output_path, "w", encoding="utf-8") as outfile:
        for result in results:
            if result:  # 정상적으로 처리된 데이터만 저장
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")  # json.dumps로 JSON 문자열로 변환
                # ensure_ascii=False로 한글(유니코드) 그대로 저장


# 메인 실행 함수
def main():
    if not os.path.exists(input_file):
        print(f"Error: 입력 파일이 존재하지 않습니다: {input_file}")
        return

    print("병렬 처리를 시작합니다...")
    asyncio.run(process_file(input_file, output_file, batch_size=5))
    print(f"작업 완료: 결과는 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
