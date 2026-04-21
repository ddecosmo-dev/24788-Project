MODEL_NAME = "allenai/Molmo2-4B" # google/gemma-4-E4B
DATASET_PATH = "coco-dataset/val2017"
ANNOTATIONS_PATH = "coco-dataset/annotations/captions_val2017.json"
RESULTS_PATH = "results/" + MODEL_NAME.split("/")[-1] + "_results.json"

MAX_LENGTH = 50