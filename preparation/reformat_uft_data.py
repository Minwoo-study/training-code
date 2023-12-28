import argparse
import json
import random
from pathlib import Path
from preprocessing_utils import setup_logging, reconstruct_command

LOG_FILENAME = "logs/preprocessing--reformat_uft_data.log"
logger = setup_logging(LOG_FILENAME)

from tqdm_logging import logging_tqdm


def list_json_files(directory):
    json_files = []
    path = Path(directory)
    for file in path.rglob('*.json'):
        json_files.append(str(file))
    return json_files


def write_to_multiple_jsonl_files(texts, base_file_name, max_lines_per_file):
    file_count = 1
    current_line = 0
    current_file = open(f"{base_file_name}_{file_count}.jsonl", 'w', encoding='utf-8')

    for text in logging_tqdm(texts, desc="Writing text"):
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
            logger.info(f"Skip writing for {text}")

    current_file.close()
    logger.info(f"Files saved in {base_file_name}_*.jsonl")


if __name__ == "__main__":
    """
    Example: python preparation/reformat_uft_data.py \
              --in_parent_path "../../dataset/unsupervised_json_final" \
              --out_parent_path "../../dataset/unsupervised_jsonl_final"
    """
    logger.info('Start preprocessing.')

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--in_parent_path', type=str, help='Parent path of the json files directory')
    parser.add_argument('--out_parent_path', type=str, help='Parent path of the jsonl files directory')
    parser.add_argument('--max_line', type=int, help='Max line per jsonl file', default=1000)

    args = parser.parse_args()

    # Reconstruct and log the run command
    run_command = reconstruct_command(args, "python preparation/reformat_sft_data.py")
    logger.info(f'Run command: {run_command}')
    
    # Initialize ouput directory
    logger.info('Setting up output directory...')
    train_path = Path(f"{args.out_parent_path}/train/")
    eval_path = Path(f"{args.out_parent_path}/eval/")
    
    if not train_path.exists():
        train_path.mkdir(parents=True)
    
    if not eval_path.exists():
        eval_path.mkdir(parents=True)

    logger.info('Setup complete.')
    logger.info(f'Training data path: {str(train_path)}')
    logger.info(f'Eval data path: {str(eval_path)}')

    # group_paths = [str(p) for p in Path(args.in_parent_path).iterdir()]
    group_paths = [
        f"{args.in_parent_path}/01.Lexis&Nexis",
        f"{args.in_parent_path}/02.Twitter",
        f"{args.in_parent_path}/03.Extra",
    ]
    
    logger.info('Start splitting data.')
    for group_path in group_paths:
        logger.info(f"Reading {group_path}...")
        all_json_files = list_json_files(group_path)

        # Read sentences
        sent_items = []
        for file_name in logging_tqdm(all_json_files, desc="Read file"):
            with open(file_name, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
                for text in data['data']:
                    if 'Raw_data' not in text or not isinstance(text['Raw_data'], str):
                        continue

                    text_clean = ' '.join(text['Raw_data'].split())
                    sent_item = {
                        'Sen_ID': text['Sen_ID'],
                        'Sentence': text_clean
                    }
                    if sent_item not in sent_items:  # avoid duplicates
                        sent_items.append(sent_item)
        
        sent_items_list = list(sent_items)

        # Split the data in each folder to train & eval
        logger.info("Split train & eval...")
        random.shuffle(sent_items_list)
        split_idx = int(len(sent_items_list) * 0.8)
        list_train = sent_items_list[:split_idx]
        list_eval = sent_items_list[split_idx:]

        # Write training text to multiple jsonl files
        dir_names = [i for i in group_path.split("/") if i != ""]
        train_file_base = f"{args.out_parent_path}/train/" + dir_names[-1]
        write_to_multiple_jsonl_files(list_train, train_file_base, args.max_line)

        # Write eval text to multiple jsonl files
        eval_file_base = f"{args.out_parent_path}/eval/" + dir_names[-1]
        write_to_multiple_jsonl_files(list_eval, eval_file_base, args.max_line)

    logger.info('Finish preprocessing.')

    logger.info('*** Data split information ***')
    logger.info('1. Training Data')
    logger.info(f'Files path: {args.out_parent_path}/train/')
    logger.info(f'Sentence num: {len(list_train)}')
    logger.info('2. Evaluation Data')
    logger.info(f'Files path: {args.out_parent_path}/eval/')
    logger.info(f'Sentence num: {len(list_eval)}')
    
    logger.info(f'Log saved in {LOG_FINENAME}')
