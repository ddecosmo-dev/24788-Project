import json
import csv
import argparse
import os
from datetime import datetime

import torch
import clip
from PIL import Image
import nltk
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

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
        
        # img_id might be a filename or integer; we store as string for dictionary keys
        res[str(img_id)] = [gen_text]
        gts[str(img_id)] = gt_captions
        
    return gts, res

def main():
    parser = argparse.ArgumentParser(description="Compute Captioning Metrics.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to input results JSON file')
    parser.add_argument('--image_dir', type=str, help='Path to clean images (required for CLIPScore)')
    parser.add_argument('--model', type=str, default='ModelX', help='Name of the model evaluated')
    parser.add_argument('--test', type=str, default='TestSet', help='Name of the test set')
    parser.add_argument('--metrics', nargs='+', 
                        choices=['cider', 'spice', 'meteor', 'clip_score'],
                        default=['cider', 'spice'],
                        help='List of metrics to run. Space separated.')
    
    args = parser.parse_args()

    # 1. Load and format data
    gts, res = load_data(args.json_path)
    image_ids = list(gts.keys())
    final_results = {img_id: {"caption": res[img_id][0]} for img_id in image_ids}

    # 2. Compute CIDEr
    if 'cider' in args.metrics:
        print("Computing CIDEr...")
        cider_scorer = Cider()
        avg, scores = cider_scorer.compute_score(gts, res)
        for i, img_id in enumerate(image_ids):
            final_results[img_id]['cider'] = scores[i]
        print(f"Average CIDEr: {avg:.4f}")

    # 3. Compute SPICE
    if 'spice' in args.metrics:
        print("Computing SPICE (requires Java)...")
        spice_scorer = Spice()
        avg, scores = spice_scorer.compute_score(gts, res)
        for i, img_id in enumerate(image_ids):
            s = scores[i]
            if isinstance(s, dict):
                score = s.get('All', {}).get('f', 0.0) if 'All' in s else s.get('test', {}).get('All', {}).get('f', 0.0)
            else:
                score = float(s)
            final_results[img_id]['spice'] = score
        print(f"Average SPICE: {avg:.4f}")

    # 4. Compute METEOR
    if 'meteor' in args.metrics:
        print("Computing METEOR...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        m_scores = []
        for img_id in tqdm(image_ids):
            hyp = res[img_id][0].split()
            refs = [ref.split() for ref in gts[img_id]]
            score = meteor_score(refs, hyp)
            final_results[img_id]['meteor'] = score
            m_scores.append(score)
        print(f"Average METEOR: {sum(m_scores)/len(m_scores):.4f}")

    # 5. Compute CLIPScore
    if 'clip_score' in args.metrics:
        if not args.image_dir:
            print("Error: --image_dir is required for CLIPScore. Skipping.")
        else:
            print("Computing CLIPScore...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            c_scores = []
            for img_id in tqdm(image_ids):
                # Handle ID to filename conversion
                filename = img_id if img_id.endswith('.jpg') else img_id.zfill(12) + ".jpg"
                img_path = os.path.join(args.image_dir, filename)
                
                try:
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    text = clip.tokenize([res[img_id][0]], truncate=True).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image)
                        text_features = model.encode_text(text)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        score = (image_features @ text_features.T).item()
                    final_results[img_id]['clip_score'] = score
                    c_scores.append(score)
                except Exception:
                    final_results[img_id]['clip_score'] = 0.0
            print(f"Average CLIPScore: {sum(c_scores)/len(c_scores):.4f}")

    # 6. Export to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/metrics-results"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{args.model}_{args.test}_{timestamp}.csv")
    
    metric_name_map = {
        'cider': 'CIDEr_Score',
        'spice': 'SPICE_Score',
        'meteor': 'METEOR_Score',
        'clip_score': 'CLIPScore_Score'
    }
    
    # Dynamically build header based on metrics selected
    header = ["Image_ID", "Generated_Caption"] + [metric_name_map[m] for m in args.metrics]
    
    print(f"Saving per-image metrics to {output_filename}...")
    with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        
        for img_id in image_ids:
            row = [img_id, final_results[img_id]['caption']]
            for m in args.metrics:
                score = final_results[img_id].get(m, 0.0)
                row.append(round(float(score), 4))
            writer.writerow(row)

    print("Done!")

if __name__ == "__main__":
    main()