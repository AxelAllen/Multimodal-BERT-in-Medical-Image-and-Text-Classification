# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application


This project is part of the Software Project "Language, Action and Perception" at University of Saarland, WS 2021.

The repository is still a work in progress as the project goes on.

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

We have created a [shared Google Drive - LAP](https://drive.google.com/drive/folders/1gwgx4ZApTKz5fN6SG9YkiVjVCZ0WNGeH?usp=sharing) 
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

## Results

We evaluated our model on 2 different labeling schemes. In our main labeling scheme we extracted the labels from the 'major' field in the radiologists report. We labeled everything tagged as 'normal' as 0 and everything else as  'abnormal' or 1. This led to approximately 40% normal cases and 60% abnormal cases. In addition, we experimented using the 'impression' field in the radiologists report as the basis for our labels, as this is likely what a medical professional would look at in determining the significance of the report.  This led to approximately 60% normal cases and 40% abnormal cases, the exact opposite of our main labeling scheme.  We used the text from the 'findings' field in the radiologists report as our text modality for all of our experiments. We then evaluated our models on both of these labeling schemes. The results can be seen in the table below.

#### Unimodal vs. Multimodal models

| Model             | MMBT (major)      | MMBT (impression) |
|-------------------|-------------------|-------------------|
| Multimodal        | 0.96              | 0.86              |
| Text-only         | 0.97              | 0.88              |
| Image-only        | 0.74              | 0.74              |


In addition, we tried a third multilabel labeling scheme, which was based on the 'major' labels. However, we created a third category for cases that were borderline abnormal and did not involve any kinds of diseases, but other abnormalities, such as medical instruments or devices. We report our multilabel results in F1 instead of accuracy. The results can be seen in the table below.

#### Findings: Binary vs. Multilabel

| Model             | MMBT - Binary (accuracy)      | MMBT - Multilabel (macro/micro F1) |
|-------------------|-------------------------------|------------------------------------|
| Multimodal        | 0.96                          | 0.82/0.93                          |
| Text-only         | 0.97                          | 0.86/0.94                          |
| Image-only        | 0.74                          | N/A                                |


## Experiments


### False Predictions

Although our Text-only model seems to perform slightly better than our multimodal model we were still curious to see if there exist some edge cases where the model actually benefits from multimodality. Indeed, we did find some of these edge cases where the text model does make a false predictions but MMBT predicts it correctly. The opposite is naturally true as well. There does exist other cases where MMBT makes a mistake and the text-model does not. This naturally follows from the fact that the text model achieves a higher accuracy. However, for this specific experiment we were more interested in the former case, where the model does benefit from multimodality in terms of making correct predictions. The results of this can be seen in the table below.


| Labeling Scheme   | Total Errors (Text Model) | Total Corrected by MMBT | Corrected False Positives | Corrected False Negatives|
|-------------------|---------------------------|-------------------------|---------------------------|--------------------------|
| Major             | 17                        | 3                       | 0                         | 3                        |
| Impression        | 70                        | 31                      | 12                        | 19                       |

### Attention


### Zero Shot


### Integrated Gradients

The integrated gradients is a way to visualize and extract explanations from the images. The basic idea behind it is that we can make use of the learned weights, 
which allow us to take the partial derivatives w/ respect to the inputs (the pixels) and visualize gradients that have highest activations with respect to some threshold value. 
The integrated gradients module is a fork from this repository <https://github.com/TianhongDai/integrated-gradient-pytorch> and it comes with an open source MIT license. 
We have slightly modified the original implementation to work with our data. For more information consult the original paper ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/pdf/1703.01365.pdf). 
Also consult the **image_submodel.ipynb** notebook for more details on how it was used in our experiment.

Otherwise, the **integrated_gradients/** directory itself can be safely ignored in terms of running the experiments.

## References

Faik Aydin, Maggie Zhang, Michelle Ananda-Rajah, and Gholamreza Haffari. 2019. Medical Multimodal Classifiers Under Scarce
Data Condition. arXiv:1902.08888 [cs, stat].

Guillem Collell, Ted Zhang, and Marie-Francine Moens. 2017. Imagined visual representations as multimodal embeddings. In
Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, AAAI’17, pages 4378–4384, San Francisco, California,
USA. AAAI Press.

Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan, Laritza Rodriguez, Sameer Antani, George R.
Thoma, and Clement J. McDonald. 2016. Preparing a collection of radiology examinations for distribution and retrieval. Journal of
the American Medical Informatics Association : JAMIA, 23(2):304–310.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding. arXiv:1810.04805 [cs].

Allyson Ettinger. 2020. What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models.
Transactions of the Association for Computational Linguistics, 8:34–48.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual Learning for Image Recognition. arXiv:1512.03385
[cs].

Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Ethan Perez, and Davide Testuggine. 2020. Supervised Multimodal
Bitransformers for Classifying Images and Text. arXiv:1909.02950 [cs, stat].

Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification. arXiv:1408.5882 [cs].

Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. 2019. Revealing the Dark Secrets of BERT. In Proceedings of
the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 4365–4374, Hong Kong, China. Association for Computational Linguistics.

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for
Vision-and-Language Tasks. arXiv:1908.02265 [cs].

Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz,
Katie Shpanskaya, Matthew P. Lungren, and Andrew Y. Ng. 2017. CheXNet: Radiologist-Level Pneumonia Detection on Chest
X-Rays with Deep Learning. arXiv:1711.05225 [cs, stat].

Claudia Schulz and Damir Juric. 2020. Can Embeddings Adequately Represent Medical Terminology? New Large-Scale Medical
Term Similarity Datasets Have the Answer! Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):8775–8782.

Francesca Strik Lievers and Bodo Winter. 2018. Sensory language across lexical categories. Lingua, 204:45–61.

Mukund Sundararajan,  Ankur Taly, Qiqi Yan. 2017. Axiomatic Attribution for Deep Networks. ArXiv:1703.01365 [Cs, Stat].
https://arxiv.org/pdf/1703.01365.pdf

Hao Tan and Mohit Bansal. 2019. LXMERT: Learning Cross-Modality Encoder Representations from Transformers.
arXiv:1908.07490 [cs].

Jesse Vig. 2019. A Multiscale Visualization of Attention in the Transformer Model. In Proceedings of the 57th Annual Meeting of
the Association for Computational Linguistics: System Demonstrations, pages 37–42, Florence, Italy. Association for
Computational Linguistics.

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. 2017. ChestX-ray8:
Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax
Diseases. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3462–3471.
