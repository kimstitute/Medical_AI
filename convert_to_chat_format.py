import json

# 입력 파일 (prompt-completion 형식)
input_file = "hsu_mdai_divided_val_data.jsonl"
output_file = "hsu_mdai_divided_val_data_chat.jsonl"


def convert_to_chat_format(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            prompt = data["prompt"]
            completion = data["completion"]

            # Chat 형식으로 변환
            chat_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": completion.strip()}
                ]
            }
            outfile.write(json.dumps(chat_data, ensure_ascii=False) + "\n")
    print(f"Converted data saved to {output_file}")


# 실행
convert_to_chat_format(input_file, output_file)
