# Using the data analysis scripts 

### Call eval_metrics.py

python model-metrics.py \ 
--json_path /home/devin_ml/work/24788-Project/results/gemma-4-E4B-it_results_.json \
--model gemma \
--test COCO_val 



# Data analysis requirements!

Want to collect and save a large amount of data concerning the models and their tests. 

### Generation Metrics 
Compare the 5 or so human captions to machine generated one. 

1. CIDEr
2. SPICE

Other metrics? See how useful these are off base. 
Can easily add more. start with these!


### Other metrics or desired info

Sampling script

Randomly samples an image from the dataset
outputs the human made and both model captions for comparison in paper

Graphs? 
Probably just ablation study stuff
Think about after first script is done


### How to save data?

1. Can just save locally, save images or call in notebook 
2. Can use WandB or Tensorboard to save and log automatically (preferred)
