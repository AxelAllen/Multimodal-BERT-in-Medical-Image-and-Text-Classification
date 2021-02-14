"""
This code is adapted from the image.py by Kiela et al. (2020) in https://github.com/facebookresearch/mmbt/blob/master/mmbt/models/image.py
and the equivalent Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be able to load saved pre-trained weights.

This class makes up the image submodule of the MMBT model.

The forward function is also modified according to the forward function of the DenseNet model listed here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out
"""
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


# mapping number of image embeddings to AdaptiveAvgPool2d output size
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

# module assumes that the directory where the saved chexnet weight is in the same level as this module
MMBT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MMBT_DIR_PARENT, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SAVED_CHEXNET = os.path.join(MODELS_DIR, "SAVED_CHEXNET.pt")


class ImageEncoderDenseNet(nn.Module):
    def __init__(self, num_image_embeds, saved_model=True, path=os.path.join(MODELS_DIR, SAVED_CHEXNET)):
        """

        :type num_image_embeds: int
        :param num_image_embeds: number of image embeddings to generate; 1-9 as they map to specific numbers of pooling
        output shape in the 'POOLING_BREAKDOWN'
        :param saved_model: True to load saved pre-trained model False to use torch pre-trained model
        :param path: path to the saved .pt model file
        """
        super().__init__()
        if saved_model:
            # loading pre-trained weight, e.g. ChexNet
            # the model here expects the weight to be regular Tensors and NOT cuda Tensor
            model = torch.load(path)
        else:
            model = torchvision.models.densenet121(pretrained=True)

        # DenseNet architecture last layer is the classifier; we only want everything before that
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        # self.model same as original DenseNet self.features part of the forward function
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds])

    def forward(self, input_modal):
        """
        B = batch
        N = number of image embeddings
        1024 DenseNet embedding size, this can be changed when instantiating MMBTconfig for modal_hidden_size

        Bx3x224x224 (this is input shape) -> Bx1024x7x7 (this is shape after DenseNet CNN layers before the last layer)
        -> Bx1024xN (this is after torch.flatten step in this function below) -> BxNx1024 (this is the shape of the
        output tensor)

        :param input_modal: image tensor
        :return:
        """
        # Bx3x224x224 -> Bx1024x7x7 -> Bx1024xN -> BxNx1024
        features = self.model(input_modal)
        out = F.relu(features, inplace=True)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()

        return out  # BxNx1024