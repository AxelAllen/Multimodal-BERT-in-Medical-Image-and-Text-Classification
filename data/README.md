# Dataset Preparation

## Current Directory File Organization:

* **data**: the current directory where the script should be executed
    * **csv**: subdirectory for train/validation/test partitions in .csv format
    * **image_labels_csv**: subdirectory for the original data files with labels, image file names, and texts
    * **json**: subdirectory for train/validation/test partitions in .jsonl format
    * **models**: saved ChexNet pre-trained weight files

* preparations.ipynb
* preparations.py
* this README.md file

## Dataset Description

The multimodal dataset we use in our project is from the National Library of Medicine, National Institutes of Health, 
Bethesda, MD, USA and can be obtained from: 
https://openi.nlm.nih.gov/faq#collection.

For more information regarding the dataset, please refer to the source below:

> Demner-Fushman D, Kohli MD, Rosenman MB, Shooshan SE, Rodriguez L, Antani S, Thoma GR, McDonald CJ.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Preparing a collection of radiology examinations for distribution and retrieval.   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;J Am Med Inform Assoc. 2016 Mar;23(2):304-10. doi: 10.1093/jamia/ocv080.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Epub 2015 Jul 1. PMID: 26133894; PMCID: PMC5009925.


The dataset is an X-ray image dataset compiled by Indiana University hospital; it contains text meta-data in the form 
of radiologist notes. The original dataset consists of 7,470 chest x-rays with 3,955 radiology reports. 
For our experiments, we only want frontal X-ray images. After filtering for frontal images that contain the desired
metadata fields, our dataset contains 3,247 samples with 'impression' metadata and 2,847 samples with 'findings' 
metadata.

In Chest X-Ray reporting, in addition to the actual X-ray image, radiologists usually report their 'read' by
describing the findings and their determination of the findings. The findings and determination are summarized in the
'findings' and 'impression' fields respectively. Note that not all data entries have metadata
for both 'findings' and 'impression' in the note.  

This is the same dataset that is used in Aydin et al. (2019) in a CNN-based 
multimodal classification task.

## Train, Validation, Test Partitions

This directory contains the script for preparing the data from .csv files to PyTorch Training, Validation, and 
Test Datasets and Dataloaders.

The partition is done to create a 60/20/20 Train/Valdiation/Test splits.

## ChexNet 14

As reported in Aydin et al. (2019), their best image submodule is based on using
ChexNet 14 (Rajpurkar, 2017) pre-trained weights, which is a DenseNet121 pre-trained model fine-tuned
on 100,000 frontal-view X-ray images with 14 diseases. 
Thus, we're using the same pre-fine-tuned weights in our model. This is also why we
only focus on frontal-view X-ray images in our experiments as well.

We obtained the saved pre-trained weights from: https://github.com/arnoweng/CheXNet/.  
The ChexNet pre-trained weights had to be processed to be compatible with the current 
PyTorch architecture. This processing step, which only has to be done once, re-saves the pre-trained weights in a 
format that can be readily loaded in the current PyTorch version (1.7).


## Bibliography

>Aydin, F., Zhang, M., Ananda-Rajah, M., & Haffari, G. (2019).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Medical Multimodal Classifiers Under Scarce Data Condition. ArXiv:1902.08888.
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1902.08888

>Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., 
> Lungren, M. P., & Ng, A. Y. (2017).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learning. ArXiv:1711.05225. http://arxiv.org/abs/1711.05225



