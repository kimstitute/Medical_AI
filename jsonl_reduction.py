import json
import os
from tqdm import tqdm  # tqdm 라이브러리 import

# 데이터 경로 설정
question_jsonl_file = "hsu_mdai_training_data_question.jsonl"
answer_jsonl_file = "hsu_mdai_training_data_answer.jsonl"

reduced_question_jsonl_file = "hsu_mdai_training_data_question_reduced.jsonl"
reduced_answer_jsonl_file = "hsu_mdai_training_data_answer_reduced_18.jsonl"

# 몇 분의 1로 줄인 것인지 선택
N_SIZE = 18

# 질문 데이터, 혹은 답변 데이터 중 어떤 것을 다룰 건지 선택
QUESTION_PROCESS = False

if QUESTION_PROCESS:
    jsonl_file = question_jsonl_file
    output_file = reduced_question_jsonl_file
else:
    jsonl_file = answer_jsonl_file
    output_file = reduced_answer_jsonl_file


# JSONL 파일을 읽고 N_SIZE개 중 한 개만 저장하여 축소시키는 함수
def process_file(jsonl_file, output_file, n_size):
    if not os.path.exists(jsonl_file):
        print(f"Error: 입력 파일이 존재하지 않습니다: {jsonl_file}")
        return

    selected_datas = []

    with open(jsonl_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()  # JSONL 파일 내용을 줄 단위로 읽어 리스트로 반환

        # tqdm으로 진행 상황 표시
        for count, line in enumerate(tqdm(lines, desc="Processing lines")):
            if count % n_size == 0:
                selected_datas.append(line)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for selected_data in selected_datas:
            outfile.write(selected_data)

    print(f"작업 완료: 결과는 {output_file}에 저장되었습니다.")


# 메인 실행 함수
def main():
    process_file(jsonl_file, output_file, N_SIZE)


if __name__ == "__main__":
    main()
