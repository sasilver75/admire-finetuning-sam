from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from transformers import TextStreamer


# Load the model using ... full precision (fp32) (FP32 (current): ~8GB)
# model, tokenizer = FastVisionModel.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     load_in_4bit=False,
#     use_gradient_checkpointing="unsloth",
# )

# Load the model using half-precision bf16 (FP16/BF16: ~4GB) (NOTE: I might be able to do Qwen2-Vl-7B-Instruct using BF16)
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    use_gradient_checkpointing="unsloth",
    dtype=torch.bfloat16,  # Load in bfloat16
)

# It looks like "We also support finetuning ONLY the vision part of the model, or ONLY the langauge part"
# Or you can select both! You can also select to finetune the attention or the MLP layers
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Leave the vision layers frozen
    finetune_language_layers=True,  # Finetune the language layers
    finetune_attention_modules=True,  # Finetune the attention modules
    finetune_mlp_modules=True,  # Finetune the MLP modules

    r=16,  # Rank of the LoRA matrices; Larger, the higher the acc, but might overfit
    lora_alpha=16,  # Recommneded alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_rslora=False,  # They support rank stabilized LoRA
    loftq_config = None,  # They also support LoftQ
    # target_modules = "all-linear"  # Optional now; can specify a list if needed
)

dataset = load_dataset("unsloth/Radiology_mini", split="train")


instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }


converted_dataset = [convert_to_conversation(sample) for sample in dataset]

print(converted_dataset[0])


# Enable the model for inference!
FastVisionModel.for_inference(model) # Enable for inference!

image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
# TextStreamer is a utility class from `transformers` that enables real-time/streaming text output during model generation
text_streamer = TextStreamer(tokenizer) # skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)


from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

# The SFT trainer is from HuggingFace's trl library
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,  # SAmples processed per GPU forward pass # Even thuogh bs=1, we accumumulate gradients over 8 steps
        gradient_accumulation_steps = 8,  # Accumulate gradients over 8 steps before updating
        warmup_steps = 5,  # Number of steps over which we gradually increase learning rate (bs*gradient_acc*warmupsteps)
        max_steps = 30,  # Total training steps (gradient updates); alternative to the 
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,  # Initial learning rate
        fp16 = not is_bf16_supported(),  # Uses 16-bit floating point if bfloat16 isn't available
        bf16 = is_bf16_supported(),  # Uses bfloat16 if supported (better numerical stability)
        logging_steps = 1,
        optim = "adamw_8bit",  # 8-bit AdamW optimizer for memory efficiency
        weight_decay = 0.01,  # L2 regularization to fight overfitting
        lr_scheduler_type = "linear",  # Learning rate decreases linearly to zero
        seed = 3407,
        output_dir = "outputs",  # Directory to save model checkpoints
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

trainer_stats = trainer.train()
