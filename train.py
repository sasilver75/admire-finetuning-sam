import tqdm
from unsloth import FastVisionModel, is_bf16_supported
import torch
from datasets import load_dataset
import random
from transformers import AutoProcessor
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from datetime import datetime
from dotenv import load_dotenv
import os
import utils

load_dotenv()
if os.getenv("HUGGINGFACE_TOKEN") is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set")
hf_token = os.environ["HUGGINGFACE_TOKEN"]


MODEL_ORG_NAME = "Qwen"
MODEL_NAME = "Qwen2-VL-2B-Instruct"


def main():
    model_name = f"{MODEL_ORG_NAME}/{MODEL_NAME}"

    # Load model 
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=False,
        # load_in_8bit=False
        dtype=torch.bfloat16,
        use_gradient_checkpointing="unsloth",
        attn_implementation="flash_attention_2",
    )

    # Load PEFT model
    print("Loading PEFT model...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    # Load dataset
    print("Loading dataset...")
    dataset_name = "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08"
    dataset = load_dataset(dataset_name, split="train")  # Load just the training split
    print(f"Dataset size: {len(dataset)}")
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)  # Use 5% of the training split as evaluation
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")


    # Convert the dataset to a user/assistant format, get the right instructions/response.
    print("Converting dataset to conversation format...")
    converted_train = []
    # for sample in tqdm.tqdm(train_dataset.select(range(100))): # TODO: REMOVE THIS SLICING!
    for sample in tqdm.tqdm(train_dataset):
        converted_train.append(utils.convert_to_conversation(sample))
    converted_eval = []
    for sample in tqdm.tqdm(eval_dataset):
        converted_eval.append(utils.convert_to_conversation(sample))

    # TODO: Test a record

    # ~~~~ TRAIN ~~~
    print("Beginning training...")
    FastVisionModel.for_training(model)  # Enable for training

    loss_callback = utils.LossCallback()

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_train,
        eval_dataset = converted_eval,
        args = SFTConfig(
            per_device_train_batch_size = 1,  # SAmples processed per GPU forward pass # Even thuogh bs=1, we accumumulate gradients over 8 steps
            gradient_accumulation_steps = 8,  # Accumulate gradients over 8 steps before updating
            warmup_steps = 5,  # Number of steps over which we gradually increase learning rate (bs*gradient_acc*warmupsteps)
            # max_steps = 30,  # Total training steps (gradient updates); alternative to the 
            num_train_epochs = 1, # Set this instead of max_steps for full training runs
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
            evaluation_strategy = "steps",
            eval_steps = 50,  # Run every n training steps
            eval_delay=0,  # Run evaluation immediately too
            per_device_eval_batch_size=1,
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
        callbacks = [loss_callback],
    )

    trainer.train()

    # TODO: Test the same record, showing how predictions changed

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    finetune_name = f"Admire-{MODEL_NAME}-Finetune-{now}"

    # Plot the loss, save the figure
    utils.plot_loss(loss_callback, finetune_name)

    # Push model directly to HuggingFace (might take a while)
    # NOTE: This took like... 15 minutes to start actually working. Weird.
    model.push_to_hub_merged(
        f"UCSC-Admire/{finetune_name}",  # Format: "org_name/repository_name"; Here, repository name isthe model name; doesn't need to exist before running this.
        tokenizer, 
        token=hf_token
    )
    print(f"Model pushed to UCSC-Admire/{finetune_name}")

if __name__ == "__main__":
    main()