import huggingface_hub
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
import os
from configs import MODEL_NAME, DATASET_PATH, ANNOTATIONS_PATH, RESULTS_PATH, MAX_LENGTH
import json
from PIL import Image
from pprint import pprint
from tqdm import tqdm

"""

This script loads the Molmo2-4B model and evaluates it on the COCO validation dataset. 
The results, including the generated captions and the ground truth captions, are saved in a JSON file.

"""

# ----- Load Image ----- #
def load_image(image_path):
    return Image.open(image_path)

# ----- Load COCO Dataset Annotations ----- #
def load_annotations(annotations_path):
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    return annotations

# ----- Map Filenames to Captions ----- #
def map_filename_to_caption(annotations):
    # Step 1: map filename to image_id
    filename_to_id = {
        img["file_name"]: img["id"]
        for img in annotations["images"]
    }

    # Step 2: map image_id to captions
    id_to_captions = {}
    for ann in annotations["annotations"]:
        img_id = ann["image_id"]
        id_to_captions.setdefault(img_id, []).append(ann["caption"])
    
    return filename_to_id, id_to_captions

# ----- Get Captions for a Given Filename ----- #
def get_captions(filename, filename_to_id, id_to_captions):
    img_id = filename_to_id.get(filename)

    if img_id is None:
        return None
    
    return id_to_captions.get(img_id, [])

# ----- Model Loading ------ #
def load_model(model_name):
    # Load the tokenizer and model from Hugging Face Hub

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True, 
        dtype="auto", 
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True,
        dtype="auto",
        device_map="auto"
    )

    return model, processor

# ----- Text Generation ------#
def generate_text(model, processor, image, prompt, max_length=50):
    # Process the image and prompt
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=prompt),
                dict(type="image", image=image),
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    # inputs = processor(image=image, text=prompt, return_tensors="pt").to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_length)
    
    # Decode the generated text
    # generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

# ----- Save Results as JSON ----- #
def save_results(results, file_path="results/results.json", suffix=""):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    full_path = file_path.replace(".json", f"_{suffix}.json")

    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)

def main():
    # load the annotations and create mappings
    annotations = load_annotations(ANNOTATIONS_PATH)
    filename_to_id, id_to_captions = map_filename_to_caption(annotations)
    
    # Load the model and processor
    model, processor = load_model(MODEL_NAME)

    results = {}

    loop = tqdm(os.listdir(DATASET_PATH), desc="Processing Images", unit="image")
    # Iterate through the dataset and generate captions
    for file in loop:
        image = load_image(os.path.join(DATASET_PATH, file))
        prompt = "Describe the image in detail."

        generated_text = generate_text(model, processor, image, prompt, MAX_LENGTH)
        # print("Generated Text:", generated_text)

        captions = get_captions(file, filename_to_id, id_to_captions)

        # Store the generated text and the ground truth captions in the results dictionary
        results[file] = {
            "generated_text": generated_text,
            "captions": captions
        }

        loop.set_postfix({"image": file})

    # Save the results to a JSON file
    save_results(results, file_path=RESULTS_PATH)

if __name__ == "__main__":
    main()