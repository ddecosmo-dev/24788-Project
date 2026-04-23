# Ablation Study Pipeline

Could be a notebook, or a py file.
Consider using WandB for sending graphs to cloud for use in report

1. Load model(s)

2. Call transformed dataset(s)

3. Call run_model.py for infernce 

4. call model-metrics.py for metrics 

5. Output metrics

Need to make scripts that. 
-------------

1. Image blurrer 
Takes the coco dataset, applies a transformation to it. (ex. 10% gaussian blur)

Saves with a relevant label, coco2017_10_blur or something 
Makes calls to data easier 

--------------