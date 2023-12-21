import csv
import json
import random
from pathlib import Path
from tqdm import tqdm


def list_json_files(directory):
    json_files = []
    path = Path(directory)
    for file in path.rglob('*.json'):
        json_files.append(str(file))
    return json_files


def list_csv_files(directory):
    json_files = []
    path = Path(directory)
    for file in path.rglob('*.csv'):
        json_files.append(str(file))
    return json_files


def write_to_multiple_txt_files(texts, base_file_name, max_lines_per_file):
    file_count = 1
    current_line = 0
    current_file = open(f"{base_file_name}_{file_count}.txt", 'w', encoding='utf-8')

    for text in tqdm(texts, desc="Writing text"):
        try:
            current_file.write(text + "\n")
            current_line += 1
            if current_line >= max_lines_per_file:
                current_file.close()
                file_count += 1
                current_file = open(f"{base_file_name}_{file_count}.txt", 'w', encoding='utf-8')
                current_line = 0
        except TypeError:
            print(f"Skip writing for {text}")

    current_file.close()
    print(f"Files saved in {base_file_name}_*.txt")


def write_to_multiple_jsonl_files(texts, base_file_name, max_lines_per_file):
    file_count = 1
    current_line = 0
    current_file = open(f"{base_file_name}_{file_count}.jsonl", 'w', encoding='utf-8')

    for text in tqdm(texts, desc="Writing text"):
        try:
            # Ensure the text is a valid JSON object, typically a dictionary.
            # If 'text' is not already a dictionary, convert or adjust it accordingly.
            json_line = json.dumps(text) + "\n"
            current_file.write(json_line)
            current_line += 1
            if current_line >= max_lines_per_file:
                current_file.close()
                file_count += 1
                current_file = open(f"{base_file_name}_{file_count}.jsonl", 'w', encoding='utf-8')
                current_line = 0
        except TypeError:
            print(f"Skip writing for {text}")

    current_file.close()
    print(f"Files saved in {base_file_name}_*.jsonl")


if __name__ == "__main__":
    out_parent_path = "../../dataset/unsupervised_jsonl"
    train_path = Path(f"{out_parent_path}/train/")
    eval_path = Path(f"{out_parent_path}/eval/")
    if not train_path.exists():
        train_path.mkdir(parents=True)
    if not eval_path.exists():
        eval_path.mkdir(parents=True)

    in_parent_path = "../../dataset/unsupervised_json"
    group_paths = [str(p) for p in Path(in_parent_path).iterdir()]

    max_lines_per_file = 1000  # Adjust this as needed

    for group_path in group_paths:
        print(f"Reading {group_path}...")
        all_json_files = list_json_files(group_path)

        # split the data in each folder to train & eval
        print("Split train & eval...")
        random.shuffle(all_json_files)
        split_idx = int(len(all_json_files) * 0.8)
        list_train = all_json_files[:split_idx]
        list_eval = all_json_files[split_idx:]

        train_texts = []
        train_ids = set()
        for file_name in tqdm(list_train, desc="Read training"):
            with open(file_name, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                for text in data['data']:
                    if 'Raw_data' not in text or not isinstance(text['Raw_data'], str):
                        continue
                    text_clean = ' '.join(text['Raw_data'].split())
                    train_texts.append({
                        'Sen_ID': text['Sen_ID'],
                        'Sentence': text_clean
                    })
                    train_ids.add(text['Sen_ID'])

        eval_texts = []
        for file_name in tqdm(list_eval, desc="Read eval"):
            with open(file_name, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                for text in data['data']:
                    if 'Raw_data' not in text or not isinstance(text['Raw_data'], str):
                        continue
                    text_clean = ' '.join(text['Raw_data'].split())
                    if text['Sen_ID'] not in train_ids:  # prevent data contamination
                        eval_texts.append({
                            'Sen_ID': text['Sen_ID'],
                            'Sentence': text_clean
                        })

        # Write training text to multiple txt files
        dir_names = [i for i in group_path.split("/") if i != ""]
        train_file_base = f"{out_parent_path}/train/" + dir_names[-1]
        write_to_multiple_jsonl_files(train_texts, train_file_base, max_lines_per_file)

        # Write eval text to multiple txt files
        eval_file_base = f"{out_parent_path}/eval/" + dir_names[-1]
        write_to_multiple_jsonl_files(eval_texts, eval_file_base, max_lines_per_file)
