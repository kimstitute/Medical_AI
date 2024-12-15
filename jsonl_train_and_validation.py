import os
import json
import random

# 입력 파일과 출력 파일 경로 설정
input_file = "hsu_mdai_training_data_question_pair_complete.jsonl"  # 원본 JSONL 파일
train_file = "hsu_mdai_divided_train_data.jsonl"  # Train 데이터 저장 파일
val_file = "hsu_mdai_divided_val_data.jsonl"  # Validation 데이터 저장 파일

# Train과 Validation 비율 설정 (예: 0.8이면 80% Train, 20% Validation)
train_ratio = 0.8

# 랜덤 시드 설정
RANDOM_SEED = 42


def split_jsonl(input_file, train_file, val_file, train_ratio=0.8, random_seed=None):
    # 원본 파일 읽기
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()  # 모든 줄 읽기 (JSONL 형식)

    # 랜덤 시드 설정 (재현성 확보)
    if random_seed is not None:
        random.seed(random_seed)

    # 데이터 섞기
    random.shuffle(lines)

    # Train과 Validation 데이터 나누기
    train_size = int(len(lines) * train_ratio)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]

    # Train 데이터 저장
    with open(train_file, "w", encoding="utf-8") as train_out:
        train_out.writelines(train_lines)

    # Validation 데이터 저장
    with open(val_file, "w", encoding="utf-8") as val_out:
        val_out.writelines(val_lines)

    print(f"Train 데이터: {len(train_lines)}개 저장 -> {train_file}")
    print(f"Validation 데이터: {len(val_lines)}개 저장 -> {val_file}")


if __name__ == "__main__":
    # 분할 실행
    if os.path.exists(input_file):
        split_jsonl(input_file, train_file, val_file, train_ratio, RANDOM_SEED)
    else:
        print(f"입력 파일이 존재하지 않습니다: {input_file}")
