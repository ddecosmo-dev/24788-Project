import json
import csv
import argparse
import os
from datetime import datetime

# Import the metrics from pycocoevalcap
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def load_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gts = {}
    res = {}
    
    for img_id, img_data in data.items():
        # Clean the generated text (removes <turn|> or extra whitespace)
        gen_text = img_data.get("generated_text", "").replace("<turn|>", "").strip()
        gt_captions = img_data.get("captions", [])
        
        # pycocoevalcap expects dict values to be lists of strings
        res[img_id] = [gen_text]
        gts[img_id] = gt_captions
        
    return gts, res

def main():
    parser = argparse.ArgumentParser(description="Compute CIDEr and SPICE for image captions.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--model', type=str, default='ModelX', help='Name of the model evaluated')
    parser.add_argument('--test', type=str, default='TestSet', help='Name of the test set')
    args = parser.parse_args()

    # 1. Load and format data
    gts, res = load_data(args.json_path)

    # 2. Compute CIDEr
    print("Computing CIDEr scores...")
    cider_scorer = Cider()
    cider_avg, cider_scores = cider_scorer.compute_score(gts, res)
    print(f"Average CIDEr: {cider_avg:.4f}")

    # 3. Compute SPICE
    print("Computing SPICE scores (this may take a moment and requires Java)...")
    spice_scorer = Spice()
    spice_avg, spice_scores = spice_scorer.compute_score(gts, res)
    print(f"Average SPICE: {spice_avg:.4f}")

    # 4. Prepare CSV Output Name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Force the output to go to your new metrics-results folder
    output_dir = "results/metrics-results"
    os.makedirs(output_dir, exist_ok=True) # Creates the folder if it doesn't exist
    output_filename = os.path.join(output_dir, f"{args.model}_{args.test}_{timestamp}.csv")
    
    # 5. Export to CSV
    image_ids = list(gts.keys())
    
    print(f"Saving per-image metrics to {output_filename}...")
    with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Write Header
        writer.writerow(["Image_ID", "Generated_Caption", "CIDEr_Score", "SPICE_Score"])
        
        for i, img_id in enumerate(image_ids):
            gen_caption = res[img_id][0]
            
            # Extract individual scores safely
            c_score = cider_scores[i]
            
            # SPICE returns a dictionary for individual scores. We extract the overall 'f' score.
            # spice_scores[i] is typically a dict like {'image_id': img_id, 'test': spice_score_dict}
            if isinstance(spice_scores[i], dict) and 'All' in spice_scores[i]:
                s_score = spice_scores[i]['All']['f']
            elif isinstance(spice_scores[i], dict) and 'test' in spice_scores[i]:
                s_score = spice_scores[i]['test']['All']['f']
            else:
                # Fallback if structure varies based on pycocoevalcap version
                s_score = spice_scores[i] if not isinstance(spice_scores[i], dict) else 0.0

            writer.writerow([img_id, gen_caption, round(c_score, 4), round(float(s_score), 4)])

    print("Done!")

if __name__ == "__main__":
    main()