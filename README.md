# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application

Part of the Software Project "Language, Action and Perception".

## Overview


### Requirements


### Instructions


## Notebooks


## Dataset

For more information regarding the dataset utilized in the project and the
preparation steps, please consult the information in the **data** directory.  

Before proceeding to train and test the model, the frontal X-ray image files used in the 
experiments can be obtained via the link to this [Shared Google Drive](https://drive.google.com/drive/folders/1VmpB1kNLESDMGL5eoglMtlsgj32zkR9P?usp=sharing).

For access to all X-ray images, include non-frontal images, please refer to
this [Other Shared Google Drive](https://drive.google.com/drive/folders/1OP6aPLMF4ib2kTCTp9YeG0b6zVVorfKW?usp=sharing).


The frontal images should be saved to the **data/NLMCXR_png_frontal** subdirectory inside
the **./data/** directory.  

The X-ray images from the other shared Google Drive should be saved to the **data/NLMCXR_png** subdirectory inside
the **./data/** directory.  

Please note that these 2 subdirectories are **NOT** included in this repo and will need
to be created as part of the preparation steps to reproduce the experiments.

## Preprocess

You can basically ignore the preprocess folder, since it's not relevant for running the any parts of the model. It includes various scripts that were used in extracting the labels from the dataset, checking that the created files match with the content of the reports and images they were created from and also filtering the frontal images from the rest.

## Integrated Gradients



## References


