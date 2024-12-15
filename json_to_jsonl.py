import json
import os
from tqdm import tqdm  # tqdm 라이브러리 import


# 데이터 경로 설정
question_data_path = r"C:\Users\PC2306\Downloads\120.초거대AI 사전학습용 헬스케어 질의응답 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL\1.질문\응급질환"
answer_data_path = r"C:\Users\PC2306\Downloads\120.초거대AI 사전학습용 헬스케어 질의응답 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL\2.답변\응급질환"

question_output_file = "hsu_mdai_training_data_question.jsonl"
answer_output_file = "hsu_mdai_training_data_answer.jsonl"

# 질문 데이터, 혹은 답변 데이터 중 어떤 것을 다룰 건지 선택
QUESTION_PROCESS = False

if QUESTION_PROCESS:
    data_path = question_data_path
    output_file = question_output_file
else:
    data_path = answer_data_path
    output_file = answer_output_file

def get_all_json_files(base_path):
    json_files = []
    folder_count = 0  # 탐색된 폴더 수
    file_count = 0    # 탐색된 파일 수

    print(f"디렉토리 탐색 시작: {base_path}")

    for root, dirs, files in os.walk(base_path):  # os.walk로 모든 하위 디렉토리 탐색
        folder_count += 1
        print(f"현재 폴더: {root} (하위 폴더 수: {len(dirs)}, 파일 수: {len(files)})")

        for file in files:
            file_count += 1
            if file.lower().endswith(".json"):  # JSON 파일만 선택
                file_path = os.path.join(root, file)
                json_files.append(file_path)

                # 선택된 JSON 파일 로그
                if len(json_files) % 1000 == 0:  # 1000개 단위로 진행 상황 출력
                    print(f"현재까지 탐색된 JSON 파일 수: {len(json_files)}")

    print(f"탐색 완료: 총 폴더 수 {folder_count}, 총 파일 수 {file_count}, JSON 파일 수 {len(json_files)}")
    return json_files


# 모든 JSON 파일 탐색
all_json_files = get_all_json_files(data_path)
print(f"탐색된 JSON 파일 수: {len(all_json_files)}")  # 탐색된 파일 수 출력
if len(all_json_files) == 0:
    print("JSON 파일이 발견되지 않았습니다. 경로와 파일 형식을 확인하세요.")
    exit()

# 질문 데이터 처리하는 코드
if QUESTION_PROCESS:
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        count = 0  # 진행 상황 추적용 카운터
        for json_file in tqdm(all_json_files, desc="Processing JSON files", unit="file"):
            try:
                if count % 1000 == 0:
                    print(f"현재 처리 중인 파일: {json_file}")  # 현재 파일 출력
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)  # JSON 로드

                # 데이터 구조 확인
                #print(f"데이터 키: {list(data.keys())}")  # JSON 데이터 키 출력

                # JSON 데이터에서 prompt 생성
                prompt = f"[Participant: {data['participantsInfo']['gender']}, {data['participantsInfo']['age']}, {data['participantsInfo']['occupation']}, {data['participantsInfo']['rPlace']}]\n"
                prompt += f"[Medhx: {data['participantsInfo']['history']}]\n"
                prompt += f"[Category: {data['disease_category']}]\n"
                prompt += f"[Disease: {data['disease_name']['kor']} ({data['disease_name']['eng']})]\n"
                prompt += f"[Intention: {data['intention']}]\n"
                prompt += f"Question: {data['question']}\n\n###"

                # completion은 비어 있는 상태로 초기화
                completion = ""

                # JSONL 형식으로 작성
                jsonl_entry = {"prompt": prompt, "completion": completion}
                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")  # UTF-8로 저장

                count += 1  # 카운트 증가

            except KeyError as e:
                print(f"KeyError 발생: {e}, 파일: {json_file}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {json_file}, 오류: {e}")
# 답변 데이터 처리하는 코드
else:
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        count = 0  # 진행 상황 추적용 카운터
        for json_file in tqdm(all_json_files, desc="Processing JSON files", unit="file"):
            try:
                if count % 1000 == 0:
                    print(f"현재 처리 중인 파일: {json_file}")  # 현재 파일 출력
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)  # JSON 로드

                # 데이터 구조 확인
                # print(f"데이터 키: {list(data.keys())}")  # JSON 데이터 키 출력

                # JSON 데이터에서 prompt 생성
                prompt = ""

                # JSON 데이터에서 completion 생성
                completion = ""
                completion += f"[Category: {data['disease_category']}]\n"
                completion += f"[Disease: {data['disease_name']['kor']} ({data['disease_name']['eng']})]\n"
                completion += f"[Department: {', '.join(data['department'])}]\n"  # 리스트를 문자열로 변환
                completion += f"[Intention: {data['intention']}]\n\n"
                completion += f"Answer: {data['answer']['intro']} "
                completion += f"{data['answer']['body']} "
                completion += f"{data['answer']['conclusion']} "

                # JSONL 형식으로 작성
                jsonl_entry = {"prompt": prompt, "completion": completion}
                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")  # UTF-8로 저장

                count += 1  # 카운트 증가

            except KeyError as e:
                print(f"KeyError 발생: {e}, 파일: {json_file}")
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {json_file}, 오류: {e}")

print(f"작업 완료: 총 {count}개의 파일을 처리했습니다.")
