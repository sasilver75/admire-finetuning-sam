import random
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import TrainerCallback
import torch
from unsloth import FastVisionModel

def convert_to_conversation(row: dict) -> dict:
    """
    Given a dictionary for a row in the datset, containing keys:
    - id
    - language
    - compound
    - sentence_type
    - sentence
    - style
    - image_1_prompt
    - image_1
    ...
    - image_5_prompt
    - image_5

    Generate a conversation turn between a user and an assitant, in which the users
    asks the assistant to rank the images from most to least relevant, based on how
    well they represent the compound (in either a literal or idiomatic sense, based
    on how it's used in the sentence).
    """
    # 1. Create list of (original_position, image) pairs
    original_images = [
        (i, row[f"image_{i}"])
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
    
    # Now the correct order is the letters assigned to positions 1,2,3,4,5
    correct_order = [original_to_letter[i] for i in range(1, 6)]

    # print(f"Original order: {original_images}\n")
    # print(f"Shuffled order: {shuffled_images}\n")
    # print(f"Shuffled order with letters: {images_with_letters}\n")
    # print(f"Original to letter: {original_to_letter}\n")
    # print(f"Correct order: {correct_order}\n")
    
    instruction = f"""You are given a compound, its use in a sentence (which determines whether a compound should be interpreted literally or idiomatically), and five images.
    The images have been given aliases of {', '.join(letters)}, respectively.
    Rank the images from most to least relevant, based on how well they represent the compound (in either a literal or idiomatic sense, based on how it's used in the sentence).
    Return the ranking of the images as a comma-separated list of the aliases, from most to least relevant.
    
    As an example, if your predicted ranking from most to least relevant is B, C, A, E, D, then you should respond with "B, C, A, E, D".
    
    <compound>
    {row["compound"]}
    </compound>
    <sentence>
    {row["sentence"]}
    </sentence>

    Given your understanding of the compound and its correct interpretation given its use in the sentence, generate the appropriate ranking of the following images:
    """

    correct_response = f"{', '.join(correct_order)}"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                *[{"type": "image", "image": img.resize((448, 448))} for _, img in shuffled_images]
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": correct_response}
            ]
        }
    ]

    return {"messages": conversation}


class LossCallback(TrainerCallback):
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []
        self.epochs = []
        self.eval_epochs = []
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of trainer initialization"""
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # If it's a training log
            if "loss" in logs:
                self.training_losses.append(logs["loss"])
                self.epochs.append(logs["epoch"])
            # If it's an eval log
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_epochs.append(logs["epoch"])

def plot_loss(loss_callback: LossCallback, finetune_name: str):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.epochs, loss_callback.training_losses, label='Training Loss')
    plt.plot(loss_callback.eval_epochs, loss_callback.eval_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/{finetune_name}.png")
    plt.close()