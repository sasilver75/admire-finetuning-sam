from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import utils
import json
import random
import traceback
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset
from PIL import Image

# Define model names
ORIGINAL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
FINETUNED_MODEL_NAME = "UCSC-Admire/Admire-Qwen2-VL-2B-Instruct-Finetune-2024-12-02_16-27-22"
DATASET_NAME = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"

def load_original_model(model_name: str) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """
    Loads the original Qwen2VL model and its processor.

    Args:
        model_name (str): Hugging Face model repository name.

    Returns:
        model: Loaded Qwen2VL model.
        processor: Corresponding processor.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return model, processor

def load_finetuned_model(model_name: str) -> Tuple[Any, AutoProcessor]:  # Any because FastVisionModel is from unsloth
    """
    Loads the finetuned model using Unsloth's FastVisionModel and its processor.

    Args:
        model_name (str): Hugging Face model repository name of the finetuned model.

    Returns:
        model: Loaded finetuned FastVisionModel.
        processor: Corresponding processor.
    """
    # FastVisionModel.from_pretrained returns (model, tokenizer)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=False,  # Adjust based on your finetuning setup
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        attn_implementation="flash_attention_2",
    )
    
    # Get the processor separately 
    # Note: The processor is actually a combination of two components: ImageProcessor and Tokenizer
    processor = AutoProcessor.from_pretrained(model_name)
    
    model.eval()  # Set to evaluation mode
    return model, processor

def convert_raw_sample(sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Image.Image], str]:
    """
    Converts a raw dataset sample into (conversation, images, expected_output) format,
    with proper shuffling of images and letter assignments.
    """
    # 1. Create list of (original_position, image) pairs
    original_images = [
        (i, sample[f"image_{i}"])
        for i in range(1, 6)
    ]
    
    # 2. Create a shuffled version of this list
    shuffled_images = original_images.copy()
    random.shuffle(shuffled_images)
    
    # 3. Assign letters A-E to the shuffled images
    letters = list("ABCDE")
    images_with_letters = [
        (letter, image)
        for letter, (_, image) in zip(letters, shuffled_images)
    ]
    
    # 4. Create mapping of original_position to assigned letter
    original_to_letter = {
        orig_pos: letter
        for (orig_pos, _), (letter, _) in zip(shuffled_images, images_with_letters)
    }
    
    # 5. Get correct order (1->5 mapped to their assigned letters)
    correct_order = [original_to_letter[i] for i in range(1, 6)]
    correct_ranking = ", ".join(correct_order)
    
    # 6. Create conversation with placeholders
    instruction = f"""You are given a compound, its use in a sentence (which determines whether a compound should be interpreted literally or idiomatically), and five images.
    The images have been given aliases of A, B, C, D, E, respectively.
    Rank the images from most to least relevant, based on how well they represent the compound (in either a literal or idiomatic sense, based on how it's used in the sentence).
    Return the ranking of the images as a comma-separated list of the aliases, from most to least relevant.
    
    As an example of what the structure of your response should look like, if your predicted ranking of the images from most to least relevant were B, C, A, E, D, then you would respond with "B, C, A, E, D".
    Do not simply copy this example; look at the provided images and determine your own correct ranking of the images.


    <compound>
    {sample['compound']}
    </compound>
    <sentence>
    {sample['sentence']}
    </sentence>

    Given your understanding of the compound and its correct interpretation given its use in the sentence, generate the appropriate ranking of the following images:
    """
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            *[{"type": "image"} for _ in range(5)]  # placeholders
        ]
    }]
    
    # Return shuffled images in A,B,C,D,E order
    shuffled_images_final = [img.resize((448, 448)) for _, img in images_with_letters]
    
    return conversation, shuffled_images_final, correct_ranking

def convert_dataset(dataset: Dataset) -> List[Tuple[List[Dict[str, Any]], List[Image.Image], str]]:
    """
    Converts the entire dataset into evaluation format, using convert_raw_sample.
    """
    converted_data = []
    for sample in dataset:
        conv, images, ranking = convert_raw_sample(sample)
        converted_data.append((conv, images, ranking))
    return converted_data


def debug_model_inputs(inputs: Dict[str, torch.Tensor]) -> None:
    """Helper function for debugging model inputs"""
    print("\nModel Input Keys:", inputs.keys())
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: shape={v.shape}, dtype={v.dtype}")

def evaluate_sample(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    conversation: List[Dict[str, Any]],
    images: List[Image.Image],
    expected_ranking: str
) -> Optional[Tuple[str, str]]:
    """
    Evaluates a single sample using the model.

    Returns:
        Tuple[str, str]: Predicted ranking, expected ranking
    """
    try:
        # 1. Format conversation with chat template
        text = processor.apply_chat_template(
            conversation, 
            tokenize=False,
            add_generation_prompt=True
        )
        print("\nChat Template Output:")
        print(text)
        
        # 2. Process inputs according to Qwen2-VL docs
        inputs = processor(
            text=[text],  # Note: text needs to be a list
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # 3. Debug model inputs
        print("\nModel Input Keys:", inputs.keys())
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"- {k}: shape={v.shape}, dtype={v.dtype}")
        
        # 4. Move to GPU and generate
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(  # Currently: T=1.0 Greedy decoding
                **inputs,
                do_sample=False,
                max_new_tokens=32,
                min_new_tokens=1,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        # 5. Decode prediction and extract just the ranking part
        predicted_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nRaw model prediction: {predicted_text}")
        
        # Extract the actual ranking (after the last "assistant" marker)
        parts = predicted_text.split("assistant")
        if len(parts) < 2:
            print("Warning: No 'assistant' marker found in output")
            return None
            
        ranking_text = parts[-1].strip()
        print(f"Extracted ranking: {ranking_text}")
        print(f"Expected ranking:  {expected_ranking.strip()}")

        
        # Validate the ranking format
        letters = set("ABCDE,")
        if not all(c in letters or c.isspace() for c in ranking_text):
            print(f"Warning: Invalid characters in ranking: {ranking_text}")
            return None
            
        return ranking_text, expected_ranking.strip()
        
    except Exception as e:
        print(f"Error processing sample: {e}")
        traceback.print_exc()
        return None

def evaluate_model(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    converted_test: List[Tuple[List[Dict[str, Any]], List[Image.Image], str]]
) -> Dict[str, float]:
    """
    Evaluates the model on the test set.
    
    Args:
        model: The Qwen2-VL model
        processor: The model's processor
        converted_test: List of (conversation, images, expected_ranking) tuples
    
    Returns:
        dict: Metrics including accuracy and correlation scores
    """
    results = []
    
    for conversation, images, expected_ranking in tqdm(converted_test):
        result = evaluate_sample(model, processor, conversation, images, expected_ranking)
        if result is not None:  # TODO: Maybe we should count the number of occurrences? What does a result of None mean here?
            results.append(result)
    
    # Calculate metrics
    total = len(results)
    if total == 0:
        return {"top1_accuracy": 0.0, "spearman_correlation": 0.0}
    
    # Calculate accuracy and correlation
    correct = sum(1 for pred, exp in results if pred.split(", ")[0] == exp.split(", ")[0])
    accuracy = correct / total
    
    # For Spearman correlation, we need to convert rankings to lists
    correlations = []
    for pred, true in results:
        # Convert to lists and debug print
        pred_list = pred.split(", ")
        true_list = true.split(", ")
        print(f"\nPredicted ranking: {pred_list}")
        print(f"True ranking:     {true_list}")
        
        # Convert letters to ranks (5 = highest rank for spearmanr)
        pred_ranks = []
        true_ranks = [5, 4, 3, 2, 1]  # Fixed ranks for true ordering
        
        # Get predicted ranks based on where each true item appears
        for item in true_list:
            pred_ranks.append(5 - pred_list.index(item))
            
        print(f"Predicted ranks: {pred_ranks}")
        print(f"True ranks:     {true_ranks}")
        
        # Calculate correlation for this sample
        correlation = spearmanr(pred_ranks, true_ranks).correlation
        print(f"Correlation:    {correlation}")
        correlations.append(correlation)
    
    avg_correlation = sum(correlations) / len(correlations)
    
    return {
        "top1_accuracy": accuracy,
        "spearman_correlation": avg_correlation
    }

def main():
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    test_dataset = test_dataset.select(range(25))  # TODO: REMOVE ME!!! FOR TEST ONLY
    print(f"Dataset size: {len(test_dataset)}")

    # Load original model and processor
    print("Loading original model...")
    original_model, original_processor = load_original_model(ORIGINAL_MODEL_NAME)
    print("Original model loaded.")

    print("Processor Configuration:")
    print(f"- Image processor parameters:")
    print(f"  - do_resize: {original_processor.image_processor.do_resize}")
    print(f"  - do_rescale: {original_processor.image_processor.do_rescale}")
    print(f"  - do_normalize: {original_processor.image_processor.do_normalize}")
    print(f"  - min_pixels: {original_processor.image_processor.min_pixels}")
    print(f"  - max_pixels: {original_processor.image_processor.max_pixels}")
    print(f"- Tokenizer parameters:")
    print(f"  - vocab_size: {original_processor.tokenizer.vocab_size}")
    print(f"  - model_max_length: {original_processor.tokenizer.model_max_length}")
    print(f"  - padding_side: {original_processor.tokenizer.padding_side}")
    print(f"  - pad_token: {original_processor.tokenizer.pad_token}")

    # Load finetuned model and processor
    # print("Loading finetuned model...")
    # finetuned_model, finetuned_processor = load_finetuned_model(FINETUNED_MODEL_NAME)
    # print("Finetuned model loaded.")

    # Convert test dataset to conversation format (if not already)
    print("Converting test dataset...")
    converted_test = []
    for sample in tqdm(test_dataset):
        conversation, images, expected_ranking = convert_raw_sample(sample)
        converted_test.append((conversation, images, expected_ranking))
    print(f"Converted test samples: {len(converted_test)}")

    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, original_processor, converted_test)
    print("Original model evaluation completed.")

    # Evaluate finetuned model
    # print("Evaluating finetuned model...")
    # finetuned_metrics = evaluate_model(finetuned_model, finetuned_processor, converted_test)
    # print("Finetuned model evaluation completed.")

    # Display Results
    print("\nEvaluation Results:")
    print("===================================")
    print("Original Model:")
    print(f"Top-1 Accuracy: {original_metrics['top1_accuracy']:.3f}")
    print(f"Spearman Correlation: {original_metrics['spearman_correlation']:.3f}")    # print("-----------------------------------")
    # print("Finetuned Model:")
    # print(f"Top-1 Accuracy: {finetuned_metrics['top1_accuracy']:.3f}")
    # print(f"Spearman Correlation: {finetuned_metrics['mean_spearman_correlation']:.3f}")
    print("===================================")

if __name__ == "__main__":
    main()