# main.py
from src.utils.data_loader import load_dataset

dataset = load_dataset("data/advbench_subset.json")
print(f"✅ Loaded {len(dataset)} samples.")
print(dataset[0])
