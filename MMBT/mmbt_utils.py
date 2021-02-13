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
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

logger = logging.get_logger(__name__)

# directories and data filenames
JSONL_DATA_DIR = 'json'
IMG_DATA_DIR = '"NLMCXR_png_frontal"'
VAL_FILE = "image_labels_both_frontal_val.jsonl"
TEST_FILE = "image_labels_both_frontal_test.jsonl"
TRAIN_FILE = "image_labels_both_frontal_train.jsonl"


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


def get_labels():
    """
    0: normal
    1: abnormal

    :return: label classes
    """

    return [0, 1]


def get_image_transforms():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )


def load_examples(tokenizer, max_seq_len, num_image_embeds, evaluate=False, test=False, data_dir=JSONL_DATA_DIR,
                  img_dir=IMG_DATA_DIR):
    if evaluate and not test:
        path = os.path.join(data_dir, VAL_FILE)
    elif evaluate and test:
        path = os.path.join(data_dir, TEST_FILE)
    elif not evaluate and not test:
        path = os.path.join(data_dir, TRAIN_FILE)
    else:
        # shouldn't get here not evaluate and test?
        raise ValueError("invalid data file option!!")

    img_transforms = get_image_transforms()
    labels = get_labels()
    dataset = JsonlDataset(path, img_dir, tokenizer, img_transforms, labels, max_seq_len - num_image_embeds - 2)
    return dataset
