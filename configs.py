MODEL_NAME = "google/gemma-4-E4B-it" # "google/gemma-4-E4B" "allenai/Molmo2-4B"
DATASET_PATH = "coco-dataset/val2017"
ANNOTATIONS_PATH = "coco-dataset/annotations/captions_val2017.json"
RESULTS_PATH = "results/" + MODEL_NAME.split("/")[-1] + "_results.json"

# for CIDEr and SPICE evaluation, need to keep output short
PROMPT = "Describe this image in one brief sentence." #"Describe the image in detail." "long caption 70: Describe everything you see in detail, including specific descriptions of spatial positioning and relationships"
MAX_NEW_TOKENS = 2048