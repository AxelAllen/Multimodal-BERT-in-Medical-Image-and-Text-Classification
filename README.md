# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application

FYI - Work in Progress.

This will be merged with other branched 'master' readme outside of subdirectory's readme.md to form part of the main
README.md file in the master branch.

TODO:

* upload runs/ directory for Tensorboard
* upload tensorboard notebook
* upload run_mmbt notebook
* upload run_text only notebook

## This Directory File Organization

FYI: this should be the en organization of the directories and files for the entire project, right?

This project repository is organized as follows:

* **Pre-Trained Multi-Modal Text and Image Classifier in Sparse Data Application**: the parent project directory
    * this README.md file
    * **data/**: contains data files subdirectories and data preparation scripts
        * **json/**
        * **csv/**
        * **image_labels_csv/**
        * **models/**
        * **NLCXR_front_png/**
    * **MMBT/**: contains MMBT model src codes and related utility functions
    * **runs/**: saved Tensorboards for displaying models' performance
    * **integrated_gradients/**: 
        * main.py what does this code do?
        * there's notebook image_submodule notebook here?
    * run_mmbt.py or notebook
    * run_text_only.py or notebook
    * image_only notebook
    * experiment_results notebook: tensorboard graphs
    
*Note:* The NLCXR_front_png directory is NOT provided; please make this directory after cloning the repo and obtain the
image files according to the instruction in the **/data** directory

## Supervised Multimodal BiTransformers for Classifying Images and Text (MMBT)

In our project, we are experimenting with the Supervised Multimodal BiTransformers for Classifying Images and Text
(MMBT) presented by Kiela et al. (2020). This is a BERT-based model that can accommodate multi-modal inputs.
The model aims to fuse the multiple modalities as the input data are introduced to the Transformers
architecture so that the model's attention mechanism can be applied to the multimodal inputs. For more information
regarding the model, please refer to the documentation in the **MMBT** directory.

## Bibliography

>Kiela, D., Bhooshan, S., Firooz, H., Perez, E., & Testuggine, D. (2020).     
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Supervised Multimodal Bitransformers for Classifying Images and Text. ArXiv:1909.02950.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1909.02950  