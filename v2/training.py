from datetime import datetime
import random
import PIL
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import wandb
from qwen_vl_utils import process_vision_info


"""
Attempting to follow tutorials:
HuggingFace: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
Phil Schmid: https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
"""

load_dotenv()
if os.getenv("HUGGINGFACE_TOKEN") is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set")
hf_token = os.environ["HUGGINGFACE_TOKEN"]

MODEL_ORG_NAME = "Qwen"
MODEL_NAME = "Qwen2-VL-7B-Instruct"
DATASET_NAME = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08" 

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

processor = None  # Have to make this a global so that collate_fn can access it; wouldn't be a problem in the notebook.


def _create_instruction(row: dict) -> str:
    """
    Returns the user instruction for the model
    """
    return f"""You are given a compound, its use in a sentence (which determines whether a compound should be interpreted literally or idiomatically), and five images.
    The images have respectively been given aliases of A, B, C, D, E.
    Rank the images from most to least relevant, based on how well they represent the compound (in either a literal or idiomatic sense, based on how it's used in the sentence).
    Return the ranking of the images as a comma-separated list of the aliases, from most to least relevant.
    
    As an example, if your predicted ranking from most to least relevant is B, C, A, E, D, then you should respond with "B, C, A, E, D".
    Make sure that all aliases are used in your ranking, and that each is used exactly once.
    A response of a single alias (e.g. "A") is NOT an acceptable response; include all 5 aliasses in relevancy order (descending).

    <compound>
    {row["compound"]}
    </compound>
    <sentence>
    {row["sentence"]}
    </sentence>

    Given your understanding of the compound and its correct interpretation given its use in the sentence, generate the appropriate ranking of these images.
    """

def _get_shuffled_images_with_ordering(images: list[PIL.Image]) -> tuple[list[PIL.Image], str]:
    """
    Receives images ordered by relevancy, shuffles them, and returns the shuffled images, 
    as well as the correct ordering of aliases to maintain the true relevancy order.
    CONFIRMED: This is working as intended (Dec 3, 11:21PM @Sam)
    # TODO: NOTE THAT I'M NOT CURRENTLY DOING ANY IMAGE RESIZING
    """
    # Numbered images
    numbered_images = [(i, images[i-1]) for i in range(1,6)]
    
    # Create a shuffled version of this list
    shuffled_images = numbered_images.copy()
    random.shuffle(shuffled_images)
    
    # Assign letters A-E to the shuffled images
    letters = list("ABCDE")
    images_with_letters = [
        (letter, image)
        for letter, (_, image) in zip(letters, shuffled_images)
    ]
    
    # Create mapping of original_position to assigned letter
    original_to_letter = {
        orig_pos: letter
        for (orig_pos, _), (letter, _) in zip(shuffled_images, images_with_letters)
    }

    # Now the correct order is the letters assigned to positions 1,2,3,4,5
    correct_order = [original_to_letter[i] for i in range(1, 6)]

    # Just remove the numbers from the shuffled images
    final_shuffled_images = [img for _, img in shuffled_images]
    correct_order_string = ", ".join(correct_order)

    return final_shuffled_images, correct_order_string


def format_data(sample: dict) -> list[dict]:
    """
    analgous to format_data in HFQwenTut: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
    Takes a row from the original dataset and formats it into the chatbot structure
    Has to return a dict because dataset.map requires a function that returns a dict.
    """
    # NOTE: Added this option to resize before going to bed
    images = [sample[f"image_{i}"].resize((384, 384)) for i in range(1, 6)]
    # images = [sample[f"image_{i}"] for i in range(1, 6)]

    shuffled_images, correct_order = _get_shuffled_images_with_ordering(images)
    instruction = _create_instruction(sample)

    # Return a single dictionary representing the conversation
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that can rank images based on their relevance to a given compound and sentence."}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in shuffled_images],
                {"type": "text", "text": instruction}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": correct_order}]
        }
    ]

def generate_text_from_sample(model, processor, sample, max_new_tokens=64, device="cuda"):
    """
    From the HF Qwen tutorial: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
    It seems that the samples that are put through here are samples that have already gone through format_data
    IE they're already structured as a conversation.

    NOTE: While the tutorial uses process_vision_info, Sonnet has convinced me that I don't have to.
    In the tutorial, they use process_visoin_info where they're working with a specific dataset format
    where imagse are stored in aprticular way (eg as paths or as encoded data needing preprocessing).
    In contrast, I'm working with PIL Image objects directly, which the Qwen2VL processor cna handle directly.
    The processor knows how to: Resize images, convert them to tensors, handle multiple images in a batch, apply any normalization.
    """
    # Include the full conversation including system message
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True
    )

    # Extract all five images from the user turn's content
    images = []
    for content in sample[1]["content"]:
        if content.get("type") == "image":
            images.append(content["image"])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=images,
        return_tensors="pt",
    ).to(device)

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return output_text[0]

def collate_fn(examples: list[dict]) -> dict:  # TODO: IS RETURN TYPE RIGHT?
    """
    Collate function to properly retrieve and batch the data during the training procedure
    Handles formatting of our dataset inputs, ensuring they're correctly structured for model
    
    Used to preprocess both training and evaluation batches, and the length of the list is determined
    by the respecitve batch size:
        -  For training: per_device_train_batch_size (1)
        -  For evaluation: per_device_eval_batch_size (1)
    Each element in the list is a list of dicts as returned from format_data
    """
    global processor
    if processor is None:
        raise ValueError("Processor is not set")

    # Since we're getting a list of conversations, we need to process each one
    formatted_texts = []
    all_images = []
    
    for conversation in examples:  # Usually just one conversation due to batch_size=1
        # Apply chat template to format the conversation
        text = processor.apply_chat_template(conversation, tokenize=False)
        formatted_texts.append(text)
        
        # Extract images from the user turn
        user_turn = [turn for turn in conversation if turn["role"] == "user"][0]
        images = [content["image"] for content in user_turn["content"] 
                 if content["type"] == "image"]
        all_images.extend(images)
    
    # Process both text and images together
    batch = processor(
        text=formatted_texts,
        images=all_images,
        return_tensors="pt",
        padding=True
    )
    
    # Create labels for training (same as input_ids but with padding masked)
    # TODO(SAM): CHECK THIS FUNCTION, IS THIS RIGHT? SEE TUTORIALS
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch

def main():
    # Load training dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Dataset size: {len(dataset)}")
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42) # Use 5% of the training split as evaluation
    print("Processing datasets...")
    # train_dataset = dataset_dict["train"].map(
    #     format_data,
    #     batched=False,
    #     # batch_size=32,
    #     num_proc=1,
    #     remove_columns=dataset_dict["train"].column_names,
    #     load_from_cache_file=False,  # Disable caching
    #     keep_in_memory=False # Not sure what this does yet, but o1 thinks it exists and matters?
    # )
    # print(f"Mapped training data")
    # eval_dataset = dataset_dict["test"].map(
    #     format_data,
    #     batched=False,
    #     # batch_size=32,
    #     num_proc=1,
    #     remove_columns=dataset_dict["test"].column_names,
    #     load_from_cache_file=False,  # Disable caching
    #     keep_in_memory=False # Not sure what this does yet, but o1 thinks it exists and matters?
    # )
    # print(f"Mapped evaluation data")
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Eval dataset size: {len(eval_dataset)}")

    # Process the datasets
    # print("Processing datasets...")
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]
    train_dataset = [format_data(sample) for sample in tqdm(train_dataset, desc="Processing train data")]
    eval_dataset = [format_data(sample) for sample in tqdm(test_dataset, desc="Processing eval data")]

    # Create Bits and Bytes Config
    # The model's weights are quantized to 4-bit, but the computations (activations, gradients) are done in bfloat16 for better numerical stability. 
    # This combination gives you both memory efficiency (from 4-bit quantization) and training stability (from bfloat16 compute).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Load the model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=f"{MODEL_ORG_NAME}/{MODEL_NAME}",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    print("Loading processor...")
    global processor
    processor = Qwen2VLProcessor.from_pretrained(
        f"{MODEL_ORG_NAME}/{MODEL_NAME}",
        min_pixels=256*28*28,  # Minimum number of pixels (256 tokens)
        max_pixels=384*28*28   # Maximum number of pixels (384 tokens instead of 1024 for memory savings)
    )

    # Configure QLoRA (Unlike standard LoRA, which reduces memory by applying a low-rank approximation,
    # takes it a step further by quantizing the weights of the LoRA adapters, leading to even lower memory requirements)
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print the trainable parameters
    peft_model.print_trainable_parameters()

    # Configure training arguments
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_args = SFTConfig(
        output_dir="qwen2-7b-instruct-trl-sft-ADMIRE",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=16,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_8bit",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="cosine_with_restarts",  # Type of learning rate scheduler
        weight_decay=0.05,
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=40,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=200,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=1,  # Maximum norm for gradient clipping
        warmup_ratio=0.1,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=True,  # Whether to push model to Hugging Face Hub
        hub_model_id=f"UCSC-Admire/Qwen2-VL-7B-Instruct-finetune-{now}",
        hub_token=hf_token,
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )
    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    # Conecting to weights and biases
    wandb.init(
        project="qwen2-7b-instruct-trl-sft",
        name=f"qwen2-7b-instruct-trl-sft-{now}",
        config=training_args
    )

    # Try generating text from a sample, does it work? Yes!
    sample = train_dataset[0]
    generated_text = generate_text_from_sample(peft_model, processor, sample)
    print(generated_text)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        # peft_config=peft_config, # It seems you can either pass the model + peft config or peft_model (Their way in the tutorial isn't obviously supported from the @HFQwen tutorial)
        tokenizer=processor.tokenizer,
    )

    # Train the model!
    print("Training the model...")
    trainer.train()

    print(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
    print("DONE!")