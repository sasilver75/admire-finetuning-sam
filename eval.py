from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import utils  # Import the utils module you're using in train.py

# TODO: Maybe it also makes sense to upload the original model to our repo?
# Or maybe we can just load it from the Qwen VL repo on HF...
original_model_name = "Qwen/Qwen2-VL-2B-Instruct"
finetuned_model_name = "UCSC-Admire/Admire-Finetune-2024-12-01_22-21-53"
dataset_name = "UCSC-Admire/idiom-dataset-30-2024-12-02_10-10-01"

def evaluate_model(model, tokenizer, dataset):
    """Evaluate a model on the dataset and return accuracy and correlation metrics."""
    all_predicted_rankings = []
    all_true_rankings = []
    correct_top1 = 0
    total_samples = 0
    
    for sample in tqdm(dataset):
        # Convert sample to conversation format (same as in training)
        conv_sample = utils.convert_to_conversation(sample)
        
        # Get scores for each image
        scores = []
        for img_path in sample['image_paths']:
            # Format prompt as in training
            prompt = conv_sample['text']  # This should be the instruction/question
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=False
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract numerical score from response
                try:
                    score = float(response.strip())  # Assuming model outputs a number
                except ValueError:
                    score = 0.0  # Fallback if parsing fails
                
                scores.append(score)
        
        # Convert scores to rankings (highest score = rank 1)
        predicted_ranking = np.argsort(np.argsort(-np.array(scores))) + 1
        
        # True ranking is just [1, 2, 3, 4, 5] since images are pre-ordered
        true_ranking = list(range(1, len(sample['image_paths']) + 1))
        
        # Check if top-1 prediction is correct
        if predicted_ranking[0] == 1:
            correct_top1 += 1
        
        all_predicted_rankings.append(predicted_ranking)
        all_true_rankings.append(true_ranking)
        total_samples += 1
    
    # Calculate metrics
    accuracy = correct_top1 / total_samples
    
    # Flatten rankings for overall Spearman correlation
    flat_predicted = np.array(all_predicted_rankings).flatten()
    flat_true = np.array(all_true_rankings).flatten()
    correlation, _ = spearmanr(flat_predicted, flat_true)
    
    return {
        "top1_accuracy": accuracy,
        "spearman_correlation": correlation
    }

def main():
    # Load dataset
    dataset_name = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"
    dataset = load_dataset(dataset_name, split="train")
    _, test_dataset = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    
    # Evaluate original model
    print("Evaluating original model...")
    original_model, original_tokenizer = load_model_and_tokenizer(original_model_name)
    original_metrics = evaluate_model(original_model, original_tokenizer, test_dataset)
    
    # Evaluate finetuned model
    print("Evaluating finetuned model...")
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(finetuned_model_name)
    finetuned_metrics = evaluate_model(finetuned_model, finetuned_tokenizer, test_dataset)
    
    # Print results
    print("\nResults:")
    print("Original Model:")
    print(f"Top-1 Accuracy: {original_metrics['top1_accuracy']:.3f}")
    print(f"Spearman Correlation: {original_metrics['spearman_correlation']:.3f}")
    
    print("\nFinetuned Model:")
    print(f"Top-1 Accuracy: {finetuned_metrics['top1_accuracy']:.3f}")
    print(f"Spearman Correlation: {finetuned_metrics['spearman_correlation']:.3f}")

if __name__ == "__main__":
    main()