"""
The classes and functions in this module are adapted from Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be albe to load saved pre-trained weights.

The forward function is also modified according to the forward function of the DenseNet model liste here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

"""
import json
import os
from collections import Counter
import logging

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


logger = logging.get_logger(__name__)


# mapping number of image embeddings to AdaptiveAvgPool2d output size
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

# module assumes that the directory where the saved chexnet weight is in the same level as this module
models_dir = "models"
saved_chexnet = "saved_chexnet.pt"


class ImageEncoderDenseNet(nn.Module):
    def __init__(self, num_image_embeds, saved_model=True, path=os.path.join(models_dir, saved_chexnet)):
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


class JsonlDataset(Dataset):
    def __init__(self, jsonl_data_path, img_dir, tokenizer, transforms, labels, max_seq_length):
        self.data = [json.loads(line) for line in open(jsonl_data_path)]
        # self.data_dir = os.path.dirname(data_path)
        self.img_data_dir = img_dir
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length

        # for image normalization for DenseNet
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        label = torch.LongTensor([self.labels.index(self.data[index]["label"])])

        image = Image.open(os.path.join(self.img_data_dir, self.data[index]["img"])).convert("RGB")
        image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update(row["label"])
        return label_freqs


def collate_fn(batch):
    """
    Specify batching for the torch Dataloader function

    :param batch: each batch of the JsonlDataset
    :return: text tensor, attention mask tensor, img tensor, modal start token, modal end token, label
    """
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor
