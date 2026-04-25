# Loading and Running `/src` Files

This guide explains the correct way to call the repository scripts in `/src`, and how to load `src` modules from notebooks or other Python files.

## Run from the repository root

Always run scripts from the project root (`/home/devin_ml/work/24788-Project`) so the module path resolves correctly.

### Run the main model script

```bash
python src/run_model.py
```

This script uses `src/configs.py` and expects the following data/results layout:
- Dataset directory: `../data/coco-dataset/val2017`
- Annotations file: `../data/coco-dataset/annotations/captions_val2017.json`
- Results output folder: `../results/model-results/`

### Compute metrics from a saved JSON file

```bash
python src/model-metrics.py \
    --json_path results/model-results/gemma-4-E4B-it_results_.json \
    --image_dir data/coco-dataset/val2017 \
    --model gemma-4b \
    --metrics cider spice meteor clip_score
```

This will compute CIDEr and SPICE from a caption results JSON file.

## Run the COCO helper scripts

### Print all captions for one random COCO image

```bash
python src/coco-tests/test-all-captions-image.py \
  --path data/coco-dataset/annotations/captions_val2017.json \
  --seed 42
```

### Print a few random COCO captions

```bash
python src/coco-tests/test-random-image-caption.py \
  --path data/coco-dataset/annotations/captions_val2017.json \
  --count 5
```

### Generating datasets with motion blur (ablation study)

# Example Bash Command for three levels of severity

To generate ablation data, in this case motion blur, the ```generate_motion_blur.py``` file can be called with flags to generate different levels of blue 

**NOTE:** The kernel size for motion blur should be odd.

```python src/create-ablation-data.py \
--input_dir data/coco-dataset/val2017 \
--output_dir data/ablation-datasets/k7 \
--type motion_blur \
--kernel_size 7 \
  #only use for testing, for full test omit 
--num_images 50 \
--seed 42
```

```python src/create-ablation-data.py \
--input_dir data/coco-dataset/val2017 \
--output_dir data/ablation-datasets/noise15 \
--type gaussian_noise \
--noise_std 15 \
--num_images 5 \
--seed 42
```



## Importing `src` modules from notebooks

When you work in `/notebooks`, add `../src` to `sys.path` before importing.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..") / "src"))

from configs import MODEL_NAME, DATASET_PATH, ANNOTATIONS_PATH, RESULTS_PATH
```

Use these values in notebook code, and keep notebook paths relative to `/notebooks`:
- Data: `../data/...`
- Results: `../results/...`

## Notes on relative paths

- `/src` scripts should use `../data/...` to reach dataset files.
- `/src` scripts should use `../results/...` to write outputs.
- `/notebooks` should also use `../data/...` and `../results/...`.
- For notebook imports, prefer adding `../src` to `sys.path`.

## Example notebook setup

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..") / "src"))

from configs import DATASET_PATH, ANNOTATIONS_PATH

print(DATASET_PATH)
print(ANNOTATIONS_PATH)
```

That is the recommended pattern for loading project code and data paths cleanly after the new directory reorganization.
