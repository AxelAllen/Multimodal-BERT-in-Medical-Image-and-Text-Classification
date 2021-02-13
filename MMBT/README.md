# Pre-Trained Multi-Modal Text and Image Classification in Sparse Medical Data Application

## This Directry File Organization

This directory contains the following python scripts:

* image.py contains the modified implementation of the ImageEncoder submodule based on DenseNet121
* mmbt.py implements Kiela et al. (2020) MMBT Model, following the original
  implementation and Huggingface's implementation.
* mmbt_config.py contains MMBTConfig class to specify MMBTModel instantiation
* mmbt_utils.py contains the JsonlDataset class to define torch Dataset, and dataset related utility functions for
  batching

## Supervised Multimodal BiTransformers for Classifying Images and Text (MMBT)

In our project, we are experiment with the Supervised Multimodal BiTransformers for Classifying Images and Text
(MMBT) presented by Kiela et al. (2020). This is a BERT-based model (Devlin et al., 2019) that can accommodate multi-modal inputs.
The model aims to fuse the multiple modalities as the input data are introduced to the Transformers
architecture so that the model's attention mechanism can be applied to the multimodal inputs. This is
based on the idea supported by other works that multimodal fusion yields better performance
than simply concatenating multimodal inputs.

This is in contrast to a baseline model to which we are comparing our work by Aydin et al. (2019). 
In the CNN-based model in Aydin et al. (2019), the CNN classifier reasons
over a concatenation of CNN features for both the pre-trained text and image submodules. We are comparing
our experiments to Aydin's (2020) work for the following reasons:

1. Similar transfer learning application of pre-trained multimodal model in binary classification on sparse medical data
2. We are able to experiment on the same dataset
3. Aydin's (2020) model provides a contrastive baseline for the multimodal concatenation approach to our
early fusion approach

The text module in MMBT is the standard BERT pre-trained model (Devlin et al., 2019); we use the `'bert-base-uncased'` 
model. The other modalities are encoded as 'embedded tokens' generated from the pre-trained image encoder submodule 
that are then projected to the same the dimensionality size as in the BERT text embeddings.

In the case of an image input modality, the image encoder is a pre-trained image
model. While the original model in Kiela et al. (2020) utilized a pre-trained ResNet51
model, we modified the image encoder submodule to be based on a pre-trained DenseNet121
model instead to be comparable to the best model used in Aydin et al. (2019). The image encoder submodule
modifies the pre-trained ResNet51/DenseNet121 to extract feautures prior to the final pooling layer as
described in Kiela et al. (2020). We also modified this image encoder submodule to accept intermediary
fine-tuned weights from the ChexNet14 Chest X-ray experiment (Rajpurkar, 2017).
  
## Bibliography 

>Aydin, F., Zhang, M., Ananda-Rajah, M., & Haffari, G. (2019).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Medical Multimodal Classifiers Under Scarce Data Condition. ArXiv:1902.08888.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1902.08888

>Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. ArXiv:1810.04805.   
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1810.04805

>Kiela, D., Bhooshan, S., Firooz, H., Perez, E., & Testuggine, D. (2020).     
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Supervised Multimodal Bitransformers for Classifying Images and Text. ArXiv:1909.02950.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1909.02950  

>Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K.,
> Lungren, M. P., & Ng, A. Y. (2017).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. ArXiv:1711.05225.  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;http://arxiv.org/abs/1711.05225



