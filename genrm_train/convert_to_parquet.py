import pandas as pd
import json
import os

def load_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]
    
def load_jsonl_folder(folder):
    data = []
    for file in os.listdir(folder):
        data.extend(load_jsonl(os.path.join(folder, file)))
    return data


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def load_json_folder(folder):
    data = []
    for file in os.listdir(folder):
        data.extend(load_json(os.path.join(folder, file)))
    return data



def convert_to_parquet(file):
    if file.endswith(".jsonl"):
        data = load_jsonl(file)
    elif file.endswith(".json"):
        data = load_json(file)
    else:
        raise ValueError(f"Unsupported file extension: {file}")
    df = pd.DataFrame(data)
    df.to_parquet(file.replace(".json", ".parquet"))


if __name__ == "__main__":
    json_file = "/Users/bytedance/Desktop/verl/genrm_train/data/skywork_deepscaler_merged_train_data.json"
    
    convert_to_parquet(json_file)
    