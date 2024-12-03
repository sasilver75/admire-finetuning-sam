# # THIS IS AN OLD ARCHIVED VERSION OF EVALUATION CODE. DON'T NEED IT.

# def evaluate_sample(model, processor, conv_sample):
#     try:
#         print("\nDEBUG - Processing Steps:")
        
#         # 1. Extract text and images
#         messages = conv_sample.get("messages", [])
#         instruction_text = None
#         images = []
        
#         for item in messages[0]["content"]:
#             if item.get("type") == "text":
#                 instruction_text = item["text"]
#             elif item.get("type") == "image":
#                 images.append(item["image"])
        
#         # 2. Format the conversation using chat template
#         conversation = [{"role": "user", "content": instruction_text}]
#         formatted_text = debug_chat_template(processor, conversation)
        
#         # 3. Process images separately first
#         image_inputs = processor.image_processor(
#             images,
#             return_tensors="pt"
#         )
        
#         # 4. Combine text and image inputs
#         inputs = processor(
#             text=formatted_text,
#             images=images,
#             return_tensors="pt",
#             padding=True,
#             add_special_tokens=True
#         )
        
#         # 5. Debug model inputs
#         debug_model_inputs(inputs)
        
#         # 6. Move to GPU
#         inputs = {k: v.cuda() for k, v in inputs.items()}
        
#         # 7. Generate response
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 do_sample=False,
#                 max_new_tokens=32,
#                 min_new_tokens=1,
#                 pad_token_id=processor.tokenizer.pad_token_id
#             )
        
#         # 8. Decode and extract rankings
#         response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"\nModel Response: {response}")
        
#         pred_letters = [x.strip() for x in response.split(",")]
#         true_letters = [x.strip() for x in messages[1]["content"][0]["text"].split(",")]
        
#         return pred_letters, true_letters

#     except Exception as e:
#         print(f"Error processing sample: {e}")
#         traceback.print_exc()  # Add this to get more detailed error info
#         return None

# def evaluate_model(model, processor, dataset):
#     """
#     Evaluates a single model on the dataset.
#     Args:
#         model: The model to evaluate.
#         processor: The corresponding processor.
#         dataset (list): The test dataset.
#     Returns:
#         dict: Metrics including Top-1 Accuracy and Spearman's Correlation.
#     """
#     correct_top1 = 0
#     total_samples = 0
#     sample_correlations = []

#     for conv_sample in tqdm(dataset, desc="Evaluating"):
#         pred_letters, true_letters = evaluate_sample(model, processor, conv_sample)
        
#         if pred_letters is None or true_letters is None:
#             continue

#         # Check for exact match (top-1 accuracy)
#         if pred_letters == true_letters:
#             correct_top1 += 1

#         # Compute Spearman correlation
#         if len(pred_letters) == 5 and len(true_letters) == 5:
#             try:
#                 pred_ranks = [ord(l) - ord('A') + 1 for l in pred_letters]
#                 true_ranks = [ord(l) - ord('A') + 1 for l in true_letters]
#                 rho, _ = spearmanr(pred_ranks, true_ranks)
#                 print(f"Successfully computed correlation: {rho}")
#                 sample_correlations.append(rho)
#             except Exception as e:
#                 print(f"Error computing Spearman's rho: {e}")
#                 sample_correlations.append(-1.0)
#         else:
#             print("Skipping correlation - invalid number of letters")
#             print(f"Expected 5 letters, got {len(pred_letters)} predicted and {len(true_letters)} true")
#             sample_correlations.append(-1.0)

#         total_samples += 1

#     mean_correlation = np.mean(sample_correlations) if sample_correlations else 0.0
#     top1_accuracy = correct_top1 / total_samples if total_samples > 0 else 0.0

#     return {
#         "top1_accuracy": top1_accuracy,
#         "mean_spearman_correlation": mean_correlation,
#         "total_samples": total_samples
#     }
