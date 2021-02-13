# LAP-final-project

MMBT: Model


* mmbt.py implements Kiela et al. (2020) Supervised Multimodal BiTransformer Model, following the original
implementation and Huggingface. 
* mmbt_config.py contains MMBTConfig class to specify MMBTModel instantiation
* mmbt_utils.py contains the modified implementatin of the ImageEncoder submodule based on DenseNet, 
  JsonlDataset class to define torch Dataset, and dataset related utility functions for batching 