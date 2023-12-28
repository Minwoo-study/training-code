import json
import pandas as pd
import argparse
from tqdm import tqdm
from preprocessing_utils import setup_logging, reconstruct_command

logger = setup_logging("logs/preprocessing--reformat_sft_data.log")

from tqdm_logging import logging_tqdm


def process_data(input_path, output_path, logger):
    print(f"Processing: {input_path}")
    df = pd.read_csv(input_path)
    data = []
    for row in logging_tqdm(df.itertuples(), desc="Read data"):
        data.append({
            'prompt': row.ID,
            'generation': row.ans_ID
        })

    with open(output_path, 'w', encoding='utf-8') as file:
        for item in logging_tqdm(data, desc="Write data"):
            json_string = json.dumps(item)
            file.write(json_string + '\n')
    
    # Log the number of data splits processed
    logger.info(f'Processed {len(data)} items from {input_path} to {output_path}')
    return len(data)


if __name__ == "__main__":
    """
    Example: python preparation/reformat_sft_data.py \
              --train_path "../../dataset/instruction/instruction_training_set_2000.csv" \
              --eval_path "../../dataset/instruction/instruction_test_set.csv" \
              --train_output "../../dataset/instruction/train.jsonl" \
              --eval_output "../../dataset/instruction/eval.jsonl"
    """
    logger.info('Start preprocessing.')

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--train_path', type=str, help='Path to the training dataset CSV')
    parser.add_argument('--eval_path', type=str, help='Path to the evaluation dataset CSV')
    parser.add_argument('--train_output', type=str, help='Output path for the processed training data')
    parser.add_argument('--eval_output', type=str, help='Output path for the processed evaluation data')

    args = parser.parse_args()

    # Reconstruct and log the run command
    run_command = reconstruct_command(args, "python preparation/reformat_sft_data.py")
    logger.info(f'Run command: {run_command}')

    # Process training data
    logger.info('Training Data: Convert csv to jsonl...')
    train_len = process_data(args.train_path, args.train_output, logger)

    # Process evaluation data
    logger.info('Evaluation Data: Convert csv to jsonl...')
    eval_len = process_data(args.eval_path, args.eval_output, logger)

    logger.info('Finish preprocessing.')

    logger.info('*** Data split information ***')
    logger.info('1. Training Data')
    logger.info(f'File path: {args.train_output}')
    logger.info(f'Data num: {train_len}')
    logger.info('2. Evaluation Data')
    logger.info(f'File path: {args.eval_output}')
    logger.info(f'Data num: {eval_len}')
    
    logger.info(f'Log saved in {"logs/preprocessing--reformat_sft_data.log"}')
