"""
This class is an adaptation of the MMBTConfig class in the Huggigface implementation of Kiela (2020) MMBT model
https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/src/transformers/models/mmbt/configuration_mmbt.py#L24

"""

import logging


logger = logging.get_logger(__name__)


class MMBTConfig(object):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.MMBTModel`. It is used to
    instantiate a MMBT model according to the specified arguments, defining the model architecture.

    """

    def __init__(self, transformer, encoder, num_labels=None, modal_hidden_size=1024):
        """
        :param transformer: underlying Transformer models
        :param encoder: pre-trained image submodule
        :param num_labels: Number of classes/labels
        :type num_labels: int
        :param modal_hidden_size: Embedding dimension of the non-text modality encoder
        e.g. 2048 for ResNet 1024 for DenseNet
        :type modal_hidden_size: int

        """
        self.transformer = transformer
        self.encoder = encoder
        self.modal_hidden_size = modal_hidden_size
        if num_labels:
            self.num_labels = num_labels