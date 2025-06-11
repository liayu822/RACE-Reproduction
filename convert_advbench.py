# scripts/convert_advbench.py

import csv
import json
import os

input_path = "datasets/advbench_subset.csv"
output_path = "data/advbench_subset.json"

def convert_csv_to_json(csv_path, json_path):
    dataset = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item = {
                "id": str(row["Original index"]),
                "instruction": row["goal"],
                "input": "",  # 沒有額外輸入
                "expected_output": row["target"],
                "category": row["category"]
            }
            dataset.append(item)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(dataset)} samples to {json_path}")

if __name__ == "__main__":
    convert_csv_to_json(input_path, output_path)
