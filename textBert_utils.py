# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19W4xdoimAJi5s6mgn1mOshVS1I5TvWMU

Functions used in the run_bert_text_only.ipynb notebook.

Parts of this code are adapted from [McCormick's and Ryan's Tutorial on BERT Fine-Tuning](http://mccormickml.com/2019/07/22/BERT-fine-tuning/) and the
Huggingface `run_mmimdb.py` script to execute the MMBT model. This code can
be accessed [here.](https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/run_mmimdb.py#L305)


"""
import logging
from collections import Counter
from MMBT.mmbt_utils import get_multiclass_labels, get_labels
import pandas as pd
import numpy as np
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score

from transformers import BertTokenizer
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


def get_train_val_test_data(wandb_config):
    """

    :param wandb_config: WandB Config Dict containing the train, val, test filepaths
    :return: train, val, test dataframes
    """
    # Load the train/val/test into a pandas dataframes.
    train = pd.read_csv(os.path.join(wandb_config.data_dir, wandb_config.train_file))
    val = pd.read_csv(os.path.join(wandb_config.data_dir, wandb_config.val_file))
    test = pd.read_csv(os.path.join(wandb_config.data_dir, wandb_config.test_file))

    print(f'Number of training sentences: {train.shape[0]:,}\n')
    print(f'Number of val sentences: {val.shape[0]:,}\n')
    print(f'Number of test sentences: {test.shape[0]:,}\n')

    return train, val, test


def tokenize_and_encode_data(sentences_iterable, tokenizer_encoder, max_sent_len, labels_iterable, multiclass=False):
    """
    Tokenize and encode input data with BERT tokenizer into BERT compatible tokens (input ids), attention
    masks, and label tensors

    :param multiclass: bool True if multilabel
    :param sentences_iterable: raw sentences
    :param tokenizer_encoder: name of BERT tokenizer
    :param max_sent_len: default = 256, but cannot exceed 512
    :param labels_iterable: class labels; array-like or iterable, but cannot be a generator
    :return: input_ids, attention masks, and label torch.Tensors
    """
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    if multiclass:
        labeling_classes = get_multiclass_labels()
        num_labels = len(labeling_classes)
        multilabels = []

    # For every sentence...
    for sent, label in zip(sentences_iterable, labels_iterable):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer_encoder.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_sent_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        # for multilabeling
        if multilabels:
            multi_label = torch.zeros(num_labels)
            multi_label[labeling_classes.index(label)] = 1
            multilabels.append(multi_label)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if multiclass:
        input_labels = torch.cat(multilabels, dim=0)
    else:
        input_labels = torch.tensor(labels_iterable)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences_iterable[0])
    print('Token IDs:', input_ids[0])
    print('Label:', input_labels[0])

    return input_ids, attention_masks, input_labels


def make_tensor_dataset(sentences_iterable, labels_iterable, wandb_config, saved_model=False, multiclass=False):
    """
    Make Torch TensorDataset

    :param multiclass: True if multilabel classification
    :param sentences_iterable:
    :param labels_iterable:
    :param wandb_config:
    :param saved_model: True if using saved BertTokenizer
    :return: TensorDataset that will be batched by the Torch DataLoader class
    """
    if saved_model:
        tokenizer = AutoTokenizer.from_pretrained(wandb_config.output_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(wandb_config.tokenizer_name, do_lower_case=True)
    input_ids, attention_masks, labels_tensors = tokenize_and_encode_data(sentences_iterable, tokenizer,
                                                                          wandb_config.max_seq_length, labels_iterable,
                                                                          multiclass)
    return TensorDataset(input_ids, attention_masks, labels_tensors)


def get_label_frequencies(labels_iterable):
    label_freqs = Counter()
    label_freqs.update(labels_iterable)
    return label_freqs


def get_multiclass_criterion(labels_iterable):
    label_freqs = get_label_frequencies()
    freqs = [label_freqs[label] for label in labels_iterable]
    label_weights = (torch.tensor(freqs, dtype=torch.float) / len(labels_iterable)) ** -1
    return nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())


def make_dataloader(dataset, wandb_config, eval=False):
    """
    The DataLoader needs to know our batch size for training, so we specify it here.
    For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.

    :param dataset: Torch TensorDataset (see make_tensor_dataset)
    :param wandb_config: WandB config dict containing batch size
    :param eval: True if batching for evaluation (use SequentialSampler) instead of False for training (use RandomSampler)
    :return: Torch DataLoader iterator
    """

    if eval:
        # For validation the order doesn't matter, so we'll just read them sequentially.
        return DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=wandb_config.eval_batch_size
        )
    else:
        # We'll take training samples in random order.
        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=wandb_config.eval_batch_size
        )


"""# Fine Tune BERT for Classification"""


def train(data_loaders_dict, wandb_config, model, criterion=None):
    """ Train the model """
    comment = f"train_{os.path.splitext(wandb_config.train_file)[0]}_{wandb_config.train_batch_size}"
    tb_writer = SummaryWriter(comment=comment)

    t_total = len(
        data_loaders_dict['train']) // wandb_config.gradient_accumulation_steps * wandb_config.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": wandb_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=wandb_config.learning_rate, eps=wandb_config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=wandb_config.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", data_loaders_dict['train_size'])
    logger.info("  Num Epochs = %d", wandb_config.num_train_epochs)
    logger.info(
        "  Total train batch size = %d",
        wandb_config.train_batch_size
        * wandb_config.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", wandb_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_eval_metric, n_no_improve = 0, 0
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    train_iterator = trange(int(wandb_config.num_train_epochs), desc="Epoch")
    set_seed(wandb_config)  # Added here for reproductibility
    train_dataloader = data_loaders_dict['train']
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Batch Iteration")
        for step, batch in enumerate(epoch_iterator):
            # each sample in batch is a tuple
            # batch is the return of the collate_fn function
            # see function definition for data tuple order
            batch = tuple(t.to(wandb_config.device) for t in batch)

            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            if wandb_config.multiclass:
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=None,
                               return_dict=True)
            else:
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

            logits = result.logits

            if wandb_config.multiclass:
                loss = criterion(logits, b_labels)
            else:
                loss = result.loss

            if wandb_config.gradient_accumulation_steps > 1:
                loss = loss / wandb_config.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % wandb_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), wandb_config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if wandb_config.logging_steps > 0 and global_step % wandb_config.logging_steps == 0:
                    logs = {}
                    if wandb_config.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if wandb_config.multiclass:
                            results = evaluate(data_loaders_dict, wandb_config, model, f"checkpoint_{global_step}",
                                               False, criterion)
                        else:
                            results = evaluate(data_loaders_dict, wandb_config, model, f"checkpoint_{global_step}")
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / wandb_config.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["training_loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if wandb_config.save_steps > 0 and global_step % wandb_config.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(wandb_config.output_dir, "checkpoint_{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                    # uncomment below to be able to save args
                    # torch.save(wandb_config, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

        if wandb_config.multiclass:
            results = evaluate(data_loaders_dict, wandb_config, model, f"epoch_{epoch}", False, criterion)
        else:
            results = evaluate(data_loaders_dict, wandb_config, model, f"epoch_{epoch}")

        if wandb_config.multiclass:
            eval_result = results["micro_f1"]
        else:
            eval_result = results["accuracy"]

        if eval_result > best_eval_metric:
            best_eval_metric = eval_result
            n_no_improve = 0
        else:
            n_no_improve += 1

        if n_no_improve > wandb_config.patience:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(data_loaders_dict, wandb_config, model, prefix="", test=False, criterion=None):
    if test:
        comment = f"test_{os.path.splitext(wandb_config.test_file)[0]}_{wandb_config.eval_batch_size}"
        tb_writer = SummaryWriter(comment=comment)

    eval_output_dir = wandb_config.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    if test:
        eval_dataloader = data_loaders_dict['test']
    else:
        eval_dataloader = data_loaders_dict['eval']

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", data_loaders_dict['test_size'] if test else data_loaders_dict['eval_size'])
    logger.info("  Batch size = %d", wandb_config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    for batch in tqdm(eval_dataloader, desc="Batch Evaluating"):
        model.eval()
        batch = tuple(t.to(wandb_config.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(wandb_config.device) for t in batch)
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            if wandb_config.multiclass:
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=None,
                               return_dict=True)
            else:
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
            logits = result.logits  # model outputs are always tuple in transformers (see doc)

            if wandb_config.multiclass:
                tmp_eval_loss = criterion(logits, b_labels)
            else:
                tmp_eval_loss = result.loss

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # Move logits and labels to CPU
        if wandb_config.multiclass:
            pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
        else:
            pred = F.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy()
        out_label_id = b_labels.detach().cpu().numpy()
        preds.append(pred)
        out_label_ids.append(out_label_id)

    eval_loss = eval_loss / nb_eval_steps

    result = {"loss": eval_loss}
    if wandb_config.multiclass:
        tgts = np.vstack(out_label_ids)
        preds = np.vstack(preds)
        result["macro_f1"] = f1_score(tgts, preds, average="macro")
        result["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        preds = [l for sl in preds for l in sl]
        out_label_ids = [l for sl in out_label_ids for l in sl]
        result["accuracy"] = accuracy_score(out_label_ids, preds)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    if not os.path.exists(output_eval_file):
        os.makedirs(os.path.join(eval_output_dir, prefix))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            if test:
                tb_writer.add_scalar(f'eval_{key}', result[key], nb_eval_steps)

    if test:
        tb_writer.close()

    return result


def set_seed(wandb_config):
    """

    :param wandb_config: WandB config dict containing seed, default = 42
    :return:
    """
    random.seed(wandb_config.seed)
    np.random.seed(wandb_config.seed)
    torch.manual_seed(wandb_config.seed)
    if wandb_config.n_gpu > 0:
        torch.cuda.manual_seed_all(wandb_config.seed)
