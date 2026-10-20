import json 

def fetch_pred_record():
    file_path = "data/beginning_data.jsonl"

    with open(file_path, "r") as f:
        lines = f.readlines()
        record = json.loads(lines[3])

    return record
