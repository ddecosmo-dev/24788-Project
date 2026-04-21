# 24788-Project

Project comparing the performance of mulit-modal models in generating image captions. 


### COCO install
Main folder is available, needs to be populated on device. 

**Install commands**
Download the 2017 Val images (approx. 1 GB)
wget http://images.cocodataset.org/zips/val2017.zip

Download the 2017 Train/Val annotations & captions (approx. 241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Download the 2014 Val images (approx. 6 GB)
wget http://images.cocodataset.org/zips/val2014.zip

Download the 2014 Train/Val annotations & captions (approx. 241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip


unzip once downloaded 

### Tasks
**NOTE!**
During testing and inference, we should randomly choose some images to compare the outputs and metrics of both models. To both visually and textually show how they caption images differently in the report. 

Could also be done for ablation study as images degrade.  

**Setup**
- [x] Inspect COCO dataset
- [x] Identify models, Gemma E4B and Molmo 4B
- [ ] Generate scripts or notebooks to create captions for images
- [ ] Do some basic prompt engineering and testing to ensure captions are consitent with desired results in COCO
- [ ] Generate script for metrics, mainly SPICE and CIDEr

**Initial Testing**
- [ ] Fine tune scripts and prompting techniques for Gemma and Molmo
- [ ] Run inference on both models with full set of images
- [ ] Find and compare base metrics for images

**Ablation Study**
- [ ] Currently gaussian blur, can consider alternatives.
- [ ] Script to apply gaussian blur to datasets and save as a new set of images, same imageID
- [ ] Define solid ranges of gaussian blur to apply, without losing entire semantic meaning. 
i.e 1%, 2%, 5%, 10% etc
- [ ] Run inference on blurred dataset and compute metrics


**After testing**
Mainly just writing the report and filling any gaps during testing. 

