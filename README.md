# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application

Part of the Software Project "Language, Action and Perception".

## Overview


### Requirements

* The project scripts were tested on Python 3.6 and 3.7. on Mac OS and Linux.
* We tested in Anaconda python virtual environment.
* Notebooks were tested on Google Colab and experiments were run on Google Colab when GPU is required.
* GPU is recommended to run the experiment. A single GPU is sufficient.
* Approximate runtime/experiment: 5-15 minutes.

### Instructions

1. Please create a virtual environment with the provided .yml file `LAP_environment.yml` 
2. Clone this repository
3. Download image files according to the instructions in the *Dataset* section.
4. The notebooks can be run in any order, with the exception that the `baseline_experiments_results.ipynb` notebook will
only reflect new runs if you run it afterward. You can view previously executed runs in that notebook, however.
5. To change hyperparameters for the text-only and MMBT notebooks, simply change the default values in the first cell
containing the Argument parser.  
   
    5.1 These notebooks can simply be run as is according to the default arguments.  
    5.2 To specify the experiment to be run, simply change the filenames of the desired datafile and specify the output
   directory, otherwise results will simply be written over the existing output directory.
   
6. Anything else?


   
## Notebooks

The notebooks in this directory contain the code to run the experiments. Please see each individual notebook for
more detailed explanations. Using Google Colab is recommended since they were created and tested on Colab and running the models without GPU can take a long time. If you have access to a GPU outside of Colab, it is possible of course to run the experiments on an environment of your choice but the notebooks cannot be guaranteed to work on every possible setting.

* **baseline_experiments_results.ipynb** shows the Tensorboard from the experiments with the textonly BERT
model and the MMBT model
  
* **image_submodel.ipynb** 

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

You can basically ignore the **preprocess/ folder**, since it's not relevant for running the any parts of the model. 
It includes various scripts that were used in extracting the labels from the dataset, checking that the created files 
match with the content of the reports and images they were created from and also filtering the frontal images from the 
rest.  

This directory will be updated with more explanations for the final submission. There are some details that
we need to work out after meeting with Aydin, F., one of the authors of the Aydin et al. (2019) paper to verify some
information regarding the dataset. 

## Integrated Gradients

The integrated gradients is a way to visualize and extract explanations from the images. The basic idea behind it is that we can make use of the learned weights, which allow us to take the partial derivatives w/ respect to the inputs (the pixels) and visualize gradients that have highest activations with respect to some threshold value. The integrated gradients module is a fork from this repository <https://github.com/TianhongDai/integrated-gradient-pytorch> and it comes with an open source MIT license. We have slightly modified the original implementation to work with our data. For more information consult the original paper ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/pdf/1703.01365.pdf) 

## References


