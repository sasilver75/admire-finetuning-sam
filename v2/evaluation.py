from datetime import datetime
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from scipy.stats import spearmanr
import PIL.Image
from typing import List, Dict, Tuple, Optional, Any
from training import format_data, generate_text_from_sample
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Constants
MODEL_ORG_NAME = "Qwen"
MODEL_NAME = "Qwen2-VL-7B-Instruct"
DATASET_NAME = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"
FINETUNED_MODEL_NAME = "UCSC-Admire/Qwen2-VL-7B-Instruct-finetune-2024-12-04_02-08-26"

def load_base_model() -> Tuple[Qwen2VLForConditionalGeneration, Qwen2VLProcessor]:
    """
    Load the base model with the same 4-bit quantization used during training
    """
    # Use the same quantization config as in training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        f"{MODEL_ORG_NAME}/{MODEL_NAME}",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    processor = Qwen2VLProcessor.from_pretrained(
        f"{MODEL_ORG_NAME}/{MODEL_NAME}",
        min_pixels=256*28*28,
        max_pixels=384*28*28
    )
    
    return model, processor

def load_finetuned_model() -> Tuple[Qwen2VLForConditionalGeneration, Qwen2VLProcessor]:
    """
    Load the finetuned model (base model + LoRA weights)
    """
    # First load the base model with quantization
    base_model, processor = load_base_model()
    
    # Load and apply the LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_MODEL_NAME,
        device_map="auto"
    )
    
    return model, processor

def evaluate_ranking(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """
    Calculate top-1 accuracy and Spearman correlation for a single prediction
    """
    try:
        # Clean and split the rankings
        pred_list = [x.strip() for x in prediction.split(",")]
        true_list = [x.strip() for x in ground_truth.split(",")]
        
        # Calculate top-1 accuracy (1.0 if first elements match, 0.0 otherwise)
        top1_accuracy = float(pred_list[0] == true_list[0])
        
        # Calculate Spearman correlation
        # Convert letters to ranks (5 = highest rank)
        pred_ranks = [5 - pred_list.index(item) for item in true_list]
        true_ranks = [5, 4, 3, 2, 1]  # Fixed ranks for true ordering
        
        correlation = spearmanr(pred_ranks, true_ranks).correlation
        
        return top1_accuracy, correlation
    
    except Exception as e:
        print(f"Error evaluating ranking: {e}")
        return 0.0, 0.0

def evaluate_model(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
    dataset: List[dict],
    desc: str = ""
) -> Dict[str, float]:
    """
    Evaluate a model on the dataset
    """
    accuracies = []
    correlations = []
    
    for sample in tqdm(dataset, desc=f"Evaluating {desc}"):
        try:
            # Format the sample to get both input and ground truth
            formatted_conversation = format_data(sample)
            
            # Extract ground truth from assistant's response
            ground_truth = formatted_conversation[-1]["content"][0]["text"]
            
            # Create input conversation (exclude assistant's turn)
            input_conversation = formatted_conversation[:-1]
            
            # Generate prediction
            prediction = generate_text_from_sample(model, processor, input_conversation)
            
            # Calculate metrics
            accuracy, correlation = evaluate_ranking(prediction, ground_truth)
            accuracies.append(accuracy)
            correlations.append(correlation)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # Calculate average metrics
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
    
    return {
        "top1_accuracy": avg_accuracy,
        "spearman_correlation": avg_correlation
    }

def main():
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(DATASET_NAME, split="test")#.select(range(20))  # Remove comment to test
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load base model
    print("Loading base model...")
    base_model, base_processor = load_base_model()
    
    # Load finetuned model
    print("Loading finetuned model...")
    finetuned_model, finetuned_processor = load_finetuned_model()
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_metrics = evaluate_model(base_model, base_processor, test_dataset, "base model")
    
    # Evaluate finetuned model
    print("\nEvaluating finetuned model...")
    finetuned_metrics = evaluate_model(finetuned_model, finetuned_processor, test_dataset, "finetuned model")
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print("Base Model:")
    print(f"Top-1 Accuracy: {base_metrics['top1_accuracy']:.4f}")
    print(f"Spearman Correlation: {base_metrics['spearman_correlation']:.4f}")
    print("-" * 50)
    print("Finetuned Model:")
    print(f"Top-1 Accuracy: {finetuned_metrics['top1_accuracy']:.4f}")
    print(f"Spearman Correlation: {finetuned_metrics['spearman_correlation']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()