import json
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    train_path = "../../dataset/instruction/instruction_training_set_2000.csv"
    eval_path = "../../dataset/instruction/instruction_test_set.csv"

    print("Processing training...")
    train_df = pd.read_csv(train_path)
    train_data = []
    for row in tqdm(train_df.itertuples(), desc="Read data"):
        train_data.append({
            'prompt': row.ID,
            'generation': row.ans_ID
        })
    
    out_file = '../../dataset/instruction/train.jsonl'
    with open(out_file, 'w', encoding='utf-8') as file:
        for item in tqdm(train_data, desc="Write data"):
            json_string = json.dumps(item)
            file.write(json_string + '\n')
    print(f'Data written to {out_file}')

    print("Processing testing...")
    eval_df = pd.read_csv(eval_path)
    eval_data = []
    for row in tqdm(eval_df.itertuples(), desc="Read data"):
        eval_data.append({
            'prompt': row.ID,
            'generation': row.ans_ID
        })
    
    out_file = '../../dataset/instruction/eval.jsonl'
    with open(out_file, 'w', encoding='utf-8') as file:
        for item in tqdm(eval_data, desc="Write data"):
            json_string = json.dumps(item)
            file.write(json_string + '\n')
    print(f'Data written to {out_file}')
