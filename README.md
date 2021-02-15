# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application


This project is part of the Software Project "Language, Action and Perception" at University of Saarland, WS 2021.

## This Directory File Organization

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
        * image.py
        * mmbt.py
        * mmbt_config.py
        * mmbt_utils.py
    * **runs/**: saved Tensorboards for displaying models' performance
    * **integrated_gradients/**:
        * **examples/**
        * **figures/**
        * **results/**
        * integrated_gradients.py
        * main.py
        * utils.py
        * visualization.py
    * *run_mmbt.ipynb* notebook
    * *run_bert_text_only.ipynb* notebook
    * *image_submodel.ipynb* notebook
    * *baseline_experiments_results.ipynb* notebook
    * textBert_utils.py src codes and utility functions for the *run_bert_text_only.ipynb* notebook
    
*Note:* The NLCXR_front_png directory is NOT provided; please make this directory after cloning the repo and obtain the
image files according to the instruction in the **/data** directory.  

*Note2:* Previous run's outputs and checkpoints are omitted due to large file size. When new experiments are run, the notebooks
will make a new directory in this parent directory. the **runs/** directory will also be updated during each experiment (text-only
and MMBT).

## Overview

This repo presents our baseline model, its submodules, and the experiments we performed to answer our research 
questions. Our main focus of the project is to try to answer the following questions:


* Can multimodal  embeddings  improve  transfer learning performance in  low-data  classification  tasks over 
  unimodal embeddings?
* Related works show improvements in classification and similarity tasks when combining visual and text modalities. 
  Does this effect carry over to the medical domain?   
* How does the multimodal representation contribute to the improved performance?
* How do different approaches to combining modalities affect performance?
    * Specifically, does having a more integrated fusion approach between the text and visual modalities perform 
      better in downstream tasks than simply concatenating vectors representing the two modalities?


The experiments we ran on the baseline model are as follows:

1. Image-only classification
2. Text-only classification
   2.1 3 text data options:  
        2.1.1 'impression' metadata text  
        2.1.2 'findings' metadata text  
        2.1.3 'both' impression and findings metadata texts
3. Multimodal classification: text and image inputs
   3.1 3 text data options:  
        3.1.1 'impression' metadata text  
        3.1.2 'findings' metadata text  
        3.1.3 'both' impression and findings metadata texts
   
In addition, we also present the *Integrated Gradient* to visualize and extract explanations from the images.


### Supervised Multimodal BiTransformers for Classifying Images and Text (MMBT)

In our project, we are experimenting with the Supervised Multimodal BiTransformers for Classifying Images and Text
(MMBT) presented by Kiela et al. (2020). This is a BERT-based model that can accommodate multi-modal inputs.
The model aims to fuse the multiple modalities as the input data are introduced to the Transformers
architecture so that the model's attention mechanism can be applied to the multimodal inputs. For more information
regarding the model, please refer to the documentation in the **MMBT** directory.

### Dataset

For more information regarding the dataset utilized in the project and the
preparation steps, please consult the information in the **data** directory.

Before proceeding to train and test the model, the frontal X-ray image files used in the
experiments can be obtained via the link to this [Shared Google Drive](https://drive.google.com/drive/folders/1d_Axy6ePY-ETJLIns67PDk1NvDyrgsMj?usp=sharing).

For access to all X-ray images, include non-frontal images, please refer to
this [Other Shared Google Drive](https://drive.google.com/drive/folders/1OP6aPLMF4ib2kTCTp9YeG0b6zVVorfKW?usp=sharing).


The frontal images should be saved to the **data/NLMCXR_png_frontal** subdirectory inside
the **./data/** directory.

The X-ray images from the other shared Google Drive should be saved to the **data/NLMCXR_png** subdirectory inside
the **./data/** directory.

Please note that these 2 subdirectories are **NOT** included in this repo and will need
to be created as part of the preparation steps to reproduce the experiments.

For reference, here's also a link to the original dataset of the 
[Indiana University Chest X-ray](https://openi.nlm.nih.gov/detailedresult?img=CXR111_IM-0076-1001&req=4).

### Requirements

* The project scripts were tested on Python 3.6 and 3.7. on Mac OS and Linux.
* We tested in Anaconda python virtual environment.
* Notebooks were tested on Google Colab and experiments were run on Google Colab when GPU is required.
* GPU is recommended to run the experiment. A single GPU is sufficient.
    * the MMBT experiments will run out of standard CPU memory in Google Colab unless GPU is available.
    * to run multiple experiments, it may be necessary to reset the runtime from time to time.
* Approximate runtime/experiment: 5-30 minutes.

### Instructions

1. Please create a virtual environment with the provided .yml file `LAP_environment.yml` 
2. Clone this repository
3. Download image files according to the instructions in the *Dataset* section.
4. The notebooks can be run in any order, with the exception that the `baseline_experiments_results.ipynb` notebook will
only reflect new runs if you run it afterward. You can view previously executed runs in that notebook, however.
5. To change hyperparameters for the text-only and MMBT notebooks, simply change the default values in the cell
containing the Argument parser.  
   
    5.1 These notebooks can simply be run as is according to the default arguments.  
    5.2 To specify the experiment to be run, simply change the filenames of the desired datafile and specify the output
   directory, otherwise results will simply be written over the existing output directory.
   
6. The loaded Tensorboard in the **baseline_experiments_results.ipynb** can be re-launched to reflect new
experiment results.
   
#### Alternative Instructions

We have created a [shared Google Drive - LAP_MMBT](https://drive.google.com/drive/folders/1gwgx4ZApTKz5fN6SG9YkiVjVCZ0WNGeH?usp=sharing) 
for this project where all the scripts, data, and Jupyter/Colab Notebooks in this repo have been uploaded. 
Simply download this drive and re-upload to your own Google Drive for testing. (i.e. you can only _view_ files
in this Drive and comment, but you cannot edit them.)

**IMPORTANT:** If you choose to test our project this way, please be aware that you may still need to update the 
path to the 'LAP' directory to reflect its location in your 'MyDrive'. Your `pwd` should always be the 'LAP'
directory for the codes to work as intended.


Caveats:We reserve the ability to modify this Drive from time to time; particularly to manage storage. 
We may delete or write over files in this shared Drive without notice and access is only provided for your convenience.

 
### Notebooks

The notebooks in this directory contain the code to run the experiments. Please see each individual notebook for
more detailed explanations. Using Google Colab is recommended since they were created and tested on Colab and running the models without GPU can take a long time. 
If you have access to a GPU outside of Colab, it is possible of course to run the experiments on an environment of your choice but the notebooks cannot be guaranteed 
to work on every possible setting.

* **baseline_experiments_results.ipynb** shows the Tensorboard from the experiments with the textonly BERT
model and the MMBT model
  

* **run_bert_text_only.ipynb** shows the end-to-end pipeline for running the text-only experiments


* **run_mmbt.ipynb* notebook** shows the end-to-end MMBT experiment pipeline
  
  
* **image_submodel.ipynb** This notebook details the Image-only model and how we obtained our results from that experiment.


### Preprocess

You can basically ignore the **preprocess/ folder**, since it's not relevant for running the any parts of the model. 
It includes various scripts that were used in extracting the labels from the dataset, checking that the created files 
match with the content of the reports and images they were created from and also filtering the frontal images from the 
rest.  

This directory will be updated with more explanations for the final submission. There are some details that
we need to work out after meeting with Aydin, F., one of the authors of the Aydin et al. (2019) paper to verify some
information regarding the dataset. 

### Integrated Gradients

The integrated gradients is a way to visualize and extract explanations from the images. The basic idea behind it is that we can make use of the learned weights, 
which allow us to take the partial derivatives w/ respect to the inputs (the pixels) and visualize gradients that have highest activations with respect to some threshold value. 
The integrated gradients module is a fork from this repository <https://github.com/TianhongDai/integrated-gradient-pytorch> and it comes with an open source MIT license. 
We have slightly modified the original implementation to work with our data. For more information consult the original paper ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/pdf/1703.01365.pdf). 
Also consult the **image_submodel.ipynb** notebook for more details on how it was used in our experiment.

Otherwise, the **integrated_gradients/** directory itself can be safely ignored in terms of running the experiments. 

## References

Aydin, F., Zhang, M., Ananda-Rajah, M., & Haffari, G. (2019). Medical Multimodal Classifiers Under Scarce Data Condition. ArXiv:1902.08888 [Cs,
Stat]. http://arxiv.org/abs/1902.08888

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., Shooshan, S. E., Rodriguez, L., Antani, S., Thoma, G. R., & McDonald, C. J. (2016). Preparing
a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association : JAMIA, 23(2),
304–310. https://doi.org/10.1093/jamia/ocv080

Hessel, J., & Lee, L. (2020). Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think! Proceedings of
the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 861–877.
https://www.aclweb.org/anthology/2020.emnlp-main.62

Kiela, D., Bhooshan, S., Firooz, H., Perez, E., & Testuggine, D. (2020). Supervised Multimodal Bitransformers for Classifying Images and Text.
ArXiv:1909.02950 [Cs, Stat]. http://arxiv.org/abs/1909.02950

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017).
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. ArXiv:1711.05225 [Cs, Stat]. http://arxiv.org/abs/1711.05225

Sundararajan, Mukund., Taly, Ankur., Yan, Qiqi. (2017). Axiomatic Attribution for Deep Networks. ArXiv:1703.01365 [Cs, Stat].
https://arxiv.org/pdf/1703.01365.pdf
