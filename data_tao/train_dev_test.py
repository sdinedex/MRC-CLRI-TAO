import json
import random
from pathlib import Path

# Define the file paths
input_file = "./MRC-CLRI-TAO/data_tao/converted_dataset_finentity_plus.jsonl"  # Replace with your dataset file
output_dir = Path("./MRC-CLRI-TAO/data_tao/finentity")  # Directory to save train, dev, test splits
output_dir.mkdir(exist_ok=True)

# Define split ratios
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

# Read the JSONL data
with open(input_file, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

# Shuffle the data to randomize
random.shuffle(data)

# Split the data
total = len(data)
train_split = int(total * train_ratio)
dev_split = train_split + int(total * dev_ratio)

train_data = data[:train_split]
dev_data = data[train_split:dev_split]
test_data = data[dev_split:]

# Save splits into separate files
def save_jsonl(data, filename):
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

save_jsonl(train_data, "train.jsonl")
save_jsonl(dev_data, "dev.jsonl")
save_jsonl(test_data, "test.jsonl")

print("Dataset successfully split and saved!")
