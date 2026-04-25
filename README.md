# 24788-Project

Project comparing the performance of mulit-modal models in generating image captions. 

### Env setup

make virtual env

install java for CIDER and SPICe metrics 

```sudo apt install openjdk-8-jre-headless```

```java -version```

Should be openjdkversion 1.8.0


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


### running models

info about run model


#### Prompt testing 

talk about prompt testing notebook

### Metrics

model-metrics.py saves CIDEr and SPICE values in a json file

``` python model-metrics.py --json_path model-results/gemma-4-E4B-it_results_.json --model gemma  --test COCO_val ```

using metrics notebook


### ablation study 

discuss how to make ablation study datasets

discuss how to run models on ablation datasets

discuss how to use the ablation analysis notebook

### Tasks
**NOTE!**
During testing and inference, we should randomly choose some images to compare the outputs and metrics of both models. To both visually and textually show how they caption images differently in the report. 

Could also be done for ablation study as images degrade.  

**Setup**
- [x] Inspect COCO dataset
- [x] Identify models, Gemma E4B and Molmo 4B
- [x] Generate scripts or notebooks to create captions for images
- [x] Do some basic prompt engineering and testing to ensure captions are consitent with desired results in COCO (Check)
- [x] Generate script for metrics, mainly SPICE and CIDEr
- [x] Research adding additional metrics, mainly METEOR or CLIPSCORE
- [x] Add parser to metrics to call out desired tests, not all 

**Initial Testing**
- [x] Fine tune scripts and prompting techniques for Gemma and Molmo
- [x] Run inference on both models with full set of images
- [x] Find and compare base metrics for images

**Ablation Study Setup**
- [x] Currently gaussian blur, can consider alternatives.
- [x] Script to apply gaussian blur to datasets and save as a new set of images, same imageID
- [x] Define solid ranges of gaussian blur to apply, without losing entire semantic meaning. 
1. Motion Blue: Kernel Sizes [7, 15, 31]
2. Gaussian Noise: STD [20, 40, 80]
Image examples in /viz 

**Model Baseline Testing**
- [ ] Run initial model testing
- [ ] Perform Data analysis
- [ ] Save figures and Results
 

**Ablation Study Testing**
- [ ] Create ablation datasets
- [ ] Run inference on ablation datasets
- [ ] Data analysis on ablation and between ablation models
- [ ] Model comparison notebook!


**Report**

