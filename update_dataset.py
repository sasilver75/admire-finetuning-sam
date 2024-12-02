from datasets import load_dataset
import utils
from datetime import datetime
import tqdm
from dotenv import load_dotenv
import os

"""
TODO: I _JUST_ pasted this.
I need to check it and then there are changes I can make to training.py
"""

load_dotenv()
if os.getenv("HUGGINGFACE_TOKEN") is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set")
hf_token = os.environ["HUGGINGFACE_TOKEN"]

def process_dataset(dataset_name: str):
    """
    Load, process, and save a dataset for reuse in training.
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    
    # Create train/test split
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    print("Converting datasets to conversation format...")
    
    # Process training data
    converted_train = []
    for sample in tqdm.tqdm(train_dataset, desc="Processing training data"):
        converted_train.append(utils.convert_to_conversation(sample))
    
    # Process evaluation data
    converted_eval = []
    for sample in tqdm.tqdm(eval_dataset, desc="Processing evaluation data"):
        converted_eval.append(utils.convert_to_conversation(sample))
    
    # Create new dataset with the processed data
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dataset_name = f"idiom-dataset-processed-{now}"
    
    # Convert to Dataset objects
    from datasets import Dataset
    processed_train = Dataset.from_list(converted_train)
    processed_eval = Dataset.from_list(converted_eval)
    
    # Push to hub
    processed_train.push_to_hub(
        f"UCSC-Admire/{new_dataset_name}", 
        split="train",
        token=hf_token
    )
    processed_eval.push_to_hub(
        f"UCSC-Admire/{new_dataset_name}", 
        split="validation",
        token=hf_token
    )
    
    print(f"Processed dataset pushed to UCSC-Admire/{new_dataset_name}")
    return f"UCSC-Admire/{new_dataset_name}"

if __name__ == "__main__":
    # Use your current dataset name
    dataset_name = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"
    new_dataset = process_dataset(dataset_name)