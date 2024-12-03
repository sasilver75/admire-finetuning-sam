from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import utils
import json

# Define model names
ORIGINAL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
FINETUNED_MODEL_NAME = "UCSC-Admire/Admire-Qwen2-VL-2B-Instruct-Finetune-2024-12-02_16-27-22"
DATASET_NAME = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"

def load_original_model(model_name):
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

def load_finetuned_model(model_name):
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

def extract_letters(text):
    # Find the last line that contains a comma-separated list of letters
    lines = text.strip().split('\n')
    for line in reversed(lines):
        if ',' in line:
            # Extract just the letters from the ranking
            letters = [l.strip() for l in line.split(',')]
            if all(len(letter) == 1 for letter in letters):
                return letters
    return []
def evaluate_model(model, processor, dataset):
    """
    Evaluates a single model on the dataset.
    Args:
        model: The model to evaluate.
        processor: The corresponding processor.
        dataset (list): The test dataset.
    Returns:
        dict: Metrics including Top-1 Accuracy and Spearman's Correlation.
    """
    correct_top1 = 0
    total_samples = 0
    sample_correlations = []

    for conv_sample in tqdm(dataset, desc="Evaluating"):
        try:
            print("\nDEBUG - Processing Steps:")
            
            # 1. Log input structure
            print("1. Input Messages Structure:")
            messages = conv_sample.get("messages", [])
            print(f"Messages: {messages}")

            # Skip malformed samples
            if len(messages) != 2:
                print("Skipping malformed sample.")
                continue

            # 2. Process the chat template
            instruction = messages[0]["content"]
            print("\n2. Chat Template Output:")
            chat_text = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                add_generation_prompt=True
            )
            print(f"Chat text: {chat_text}")

            # 3. Process inputs
            images = [msg["content"] for msg in messages if isinstance(msg.get("content"), (list, dict)) and any(item.get("type") == "image" for item in msg["content"])]
            inputs = processor(
                text=[chat_text],
                images=images,
                padding=True,
                return_tensors="pt"
            )
            print("\n3. Processor Outputs:")
            print(f"Input IDs shape: {inputs.input_ids.shape}")
            print(f"Pixel values shape: {inputs.pixel_values.shape}")

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model.cuda()

            # 4. Generate and process output
            print("\n4. Model Output Analysis:")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
            
            output_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Raw prediction: {output_text}")

            # Extract predicted and true rankings
            true_letters = conv_sample.get("target", "").strip().split(", ")
            
            # Find the last line containing a comma-separated list
            lines = output_text.strip().split('\n')
            pred_letters = []
            for line in reversed(lines):
                if ',' in line:
                    # Extract and validate the letters
                    candidate_letters = [l.strip() for l in line.split(',')]
                    if all(len(letter) == 1 for letter in candidate_letters):
                        pred_letters = candidate_letters
                        break

            print(f"Extracted letters: {pred_letters} (length: {len(pred_letters)})")
            print(f"True letters: {true_letters} (length: {len(true_letters)})")

            # Check for exact match (top-1 accuracy)
            if pred_letters == true_letters:
                correct_top1 += 1

            # 5. Compute Spearman correlation
            print("\n5. Correlation Calculation:")
            if len(pred_letters) == 5 and len(true_letters) == 5:
                try:
                    pred_ranks = [ord(l) - ord('A') + 1 for l in pred_letters]
                    true_ranks = [ord(l) - ord('A') + 1 for l in true_letters]
                    rho, _ = spearmanr(pred_ranks, true_ranks)
                    print(f"Successfully computed correlation: {rho}")
                    sample_correlations.append(rho)
                except Exception as e:
                    print(f"Error computing Spearman's rho: {e}")
                    sample_correlations.append(-1.0)
            else:
                print("Skipping correlation - invalid number of letters")
                print(f"Expected 5 letters, got {len(pred_letters)} predicted and {len(true_letters)} true")
                sample_correlations.append(-1.0)

            total_samples += 1

        except Exception as e:
            print(f"\nError processing sample: {e}")
            continue

    mean_correlation = np.mean(sample_correlations) if sample_correlations else 0.0
    top1_accuracy = correct_top1 / total_samples if total_samples > 0 else 0.0

    return {
        "top1_accuracy": top1_accuracy,
        "mean_spearman_correlation": mean_correlation,
        "total_samples": total_samples
    }

def main():
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(DATASET_NAME, split="test")
    test_dataset = test_dataset.select(range(10))  # TODO: REMOVE ME!!! FOR TEST ONLY
    print(f"Dataset size: {len(test_dataset)}")

    # Load original model and processor
    print("Loading original model...")
    original_model, original_processor = load_original_model(ORIGINAL_MODEL_NAME)
    print("Original model loaded.")

    # Load finetuned model and processor
    print("Loading finetuned model...")
    finetuned_model, finetuned_processor = load_finetuned_model(FINETUNED_MODEL_NAME)
    print("Finetuned model loaded.")

    # Convert test dataset to conversation format (if not already)
    print("Converting test dataset...")
    converted_test = []
    for sample in tqdm(test_dataset, desc="Converting"):
        converted_sample = utils.convert_to_conversation(sample)
        # Ensure converted_sample has messages
        if "messages" in converted_sample and converted_sample["messages"]:
            converted_test.append(converted_sample)
    print(f"Converted test samples: {len(converted_test)}")

    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, original_processor, converted_test)
    print("Original model evaluation completed.")

    # Evaluate finetuned model
    print("Evaluating finetuned model...")
    finetuned_metrics = evaluate_model(finetuned_model, finetuned_processor, converted_test)
    print("Finetuned model evaluation completed.")

    # Display Results
    print("\nEvaluation Results:")
    print("===================================")
    print("Original Model:")
    print(f"Top-1 Accuracy: {original_metrics['top1_accuracy']:.3f}")
    print(f"Spearman Correlation: {original_metrics['mean_spearman_correlation']:.3f}")
    print("-----------------------------------")
    print("Finetuned Model:")
    print(f"Top-1 Accuracy: {finetuned_metrics['top1_accuracy']:.3f}")
    print(f"Spearman Correlation: {finetuned_metrics['mean_spearman_correlation']:.3f}")
    print("===================================")

if __name__ == "__main__":
    main()