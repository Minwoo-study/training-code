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


def write_to_multiple_files(texts, base_file_name, max_lines_per_file):
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


group_paths = [
    "../../dataset/json_to_csv/article",
    "../../dataset/json_to_csv/terkinni",
    "../../dataset/json_to_csv/wiki",
    "../../dataset/json_to_csv/twitter",
]

max_lines_per_file = 1000  # Adjust this as needed

for group_path in group_paths:
    all_csv_files = list_csv_files(group_path)

    print("Split train & eval...")
    random.shuffle(all_csv_files)
    split_idx = int(len(all_csv_files) * 0.8)
    list_train = all_csv_files[:split_idx]
    list_eval = all_csv_files[split_idx:]

    train_texts = set()
    for file_name in tqdm(list_train, desc="Read training"):
        with open(file_name, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header
            for row in csv_reader:
                train_texts.add(row[10])

    eval_texts = set()
    for file_name in tqdm(list_eval, desc="Read eval"):
        with open(file_name, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header
            for row in csv_reader:
                eval_texts.add(row[10])

    # Write training text to multiple txt files
    dir_names = [i for i in group_path.split("/") if i != ""]
    train_file_base = "../../dataset/unsupervised_txt/train/" + dir_names[-1]
    write_to_multiple_files(train_texts, train_file_base, max_lines_per_file)

    # Write eval text to multiple txt files
    eval_file_base = "../../dataset/unsupervised_txt/eval/" + dir_names[-1]
    write_to_multiple_files(eval_texts, eval_file_base, max_lines_per_file)
