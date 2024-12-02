import random
from PIL import Image

def convert_to_conversation(sample: dict) -> dict:
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
    
    As an example, if your predicted ranking from most to least relevant is B, C, A, E, D, then you should respond with "B, C, A, E, D"."""

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
