import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
if os.environ.get("HUGGINGFACE_TOKEN") is None:
    raise ValueError("HUGGINGFACE_TOKEN not set")

# First, let's try to download the dataset from HuggingFace
# We don't need to use a token, because it's a public dataset.
dataset_name = "UCSC-Admire/idiom-dataset-100-2024-11-11_14-37-58"
dataset = load_dataset(dataset_name, split="train")

for sample in dataset[:3]:
    print(sample)

