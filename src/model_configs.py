MODEL_NAME = "google/gemma-4-E4B-it" # "google/gemma-4-E4B-it" "allenai/Molmo2-4B"
DATASET_PATH = "../data/ablation-datasets/noise160" # "../data/coco-dataset/val2017/"
ANNOTATIONS_PATH = "../data/coco-dataset/annotations/captions_val2017.json"
RESULTS_PATH = "../results/ablation-results/" + MODEL_NAME.split("/")[-1] + "_results.json"

# for CIDEr and SPICE evaluation, need to keep output short
PROMPT = "Describe the image in one short sentence. Ignore graininess in description. No extra details. 10 words max." #"Describe the image in detail." "long caption 70: Describe everything you see in detail, including specific descriptions of spatial positioning and relationships"
MAX_NEW_TOKENS = 2048