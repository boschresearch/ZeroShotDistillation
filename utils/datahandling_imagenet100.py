"""
    Functions for handling ImageNet and synthetic ImageNet data (combined as they are structurally similar).
"""
import argparse
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import os
import sys
import random
import time
import torch.multiprocessing as mp
import numpy as np

from utils.CustomImageFolder import ImageCaptionFolder
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    CenterCrop,
)
from torchvision.utils import make_grid
from torchvision import models
from torch import nn
from torch.nn import functional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from numpy.linalg import norm
from torchmetrics import Accuracy
from uuid import uuid4
from torch import autocast
from scipy.stats import entropy

# get dataset for synthetic data
def get_dataset(
    args,
    path,
    is_train=False,
    class_ids=[0, 1],
    id_offset=0,
    norm=None,
):
    """Function to obtain the training and validation data. For training use synthetic data, for validation ImageNet (or subset thereof)."""
    index_transform = lambda x: {
        class_ids[i]: i + id_offset for i in range(len(class_ids))
    }[x]

    # Image transformation taken from here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
    if args.test!=None:
        print("Using no augmentations")
        preprocess = Compose(
                        [
                            Resize(224),
                            CenterCrop(224),
                            ToTensor(),
                            # Choose between ImageNet and CLIP normalization, both work well: https://github.com/openai/CLIP/issues/20
                            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ]
                    )
    else:
        print("Using augmentations")
        transforms = []
        transforms.append(RandomResizedCrop(224))
        transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())

    if args.normalize_channels:
        transforms.append(Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]))

    # transform = Compose(transforms)
    preprocess = Compose(transforms)

    if is_train:
        if eval(args.precompute_teacher):
            dataset = ImageCaptionFolderPrecomputed(
                path,
                transform=preprocess,
                target_transform=index_transform,
            ) 
        else:
            dataset = ImageCaptionFolder(
                path,
                transform=preprocess,
                target_transform=index_transform,
            )
    else:
        dataset = ImageFolder(
            path,
            transform=preprocess,
            target_transform=index_transform,
        )
    dataset.samples = [s_i for s_i in dataset.samples if s_i[1] in class_ids]

    return dataset

# get concatenated dataset for synthetic data
def get_concat_dataset(
    args,
    is_train=False,
):
    """Concat train and validation data in the correct order."""
    if is_train:
        norms = args.train_norm
        class_ids = args.train_class_ids
        paths = args.train
    else:
        norms = args.val_norm
        class_ids = args.val_class_ids
        paths = args.val

    datasets = []
    for sub_dataset_idx, path in enumerate(paths):
        class_ids_sublist = (
            class_ids[sub_dataset_idx] if len(class_ids) > sub_dataset_idx else [0, 1]
        )
        norm = (
            norms[sub_dataset_idx] if norms and len(norms) > sub_dataset_idx else None
        )
        id_offset = len(
            [
                id
                for sublist_idx in range(sub_dataset_idx)
                for id in class_ids[sublist_idx]
            ]
        )  # how many class id's have been contained in the previous sub datasets
        dataset = get_dataset(
            is_train=is_train,
            args=args,
            path=path,
            class_ids=class_ids_sublist,
            norm=norm,
            id_offset=id_offset,
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)

def get_train_data_splits(args):
    """If required split the train data to construct test set from train."""
    training_dataset = get_concat_dataset(is_train=True, args=args)
    train_len = (
        args.n_train_samples if args.n_train_samples else len(training_dataset)
    ) - args.n_test_samples_from_train
    trash_len = (
        (len(training_dataset) - args.n_train_samples) if args.n_train_samples else 0
    )
    torch.manual_seed(torch.initial_seed())
    train_split, generated_train, trash = random_split(
        training_dataset, [train_len, args.n_test_samples_from_train, trash_len]
    )

    return train_split, generated_train

def get_dataloaders(args):
    """Construct the dataloader for the synthetic data."""
    train_split, generated_train = get_train_data_splits(args)
    #print(train_split)
    val_dataset = get_concat_dataset(args=args)

    if args.test!=None:
        print("Using no augmentations")
        preprocess = Compose(
                        [
                            Resize(256),
                            CenterCrop(224),
                            ToTensor(),
                            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ]
                    )
    else:
        print("Using augmentations")
        transforms = []
        transforms.append(RandomResizedCrop(224))
        transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())

        if args.normalize_channels:
            transforms.append(Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=0.26862954, 0.26130258, 0.27577711]))

        # transform = Compose(transforms)
        preprocess = Compose(transforms)
    
    if eval(args.precompute_teacher):
        dataset_train = ImageCaptionFolderPrecomputed(
            args.train[0],
            transform=preprocess,
        )
    else:
        dataset_train = ImageCaptionFolder(
            args.train[0],
            transform=preprocess,
        )
    # initialise dataloaders
    dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    generated_train_dataloader = DataLoader(generated_train, batch_size=args.batch_size,num_workers=args.num_workers)

    return dataloader, val_dataloader, generated_train_dataloader

# get dataset for real (non-synthetic) data
def get_dataset_real(
    args,
    path,
    is_train=False,
    class_ids=[0, 1],
    id_offset=0,
    norm=None,
):
    """Function to obtain the ImageNet data based on the parameters given in args."""
    index_transform = lambda x: {
        class_ids[i]: i + id_offset for i in range(len(class_ids))
    }[x]

    # Image transformation taken from here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
    if args.test!=None:
        print("Using no augmentations")
        preprocess = Compose(
                        [
                            Resize(224),
                            CenterCrop(224),
                            ToTensor(),
                            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ]
                    )
    else:
        print("Using augmentations")
        transforms = []
        transforms.append(Resize(256))
        transforms.append(CenterCrop(224))
        transforms.append(ToTensor())

        if args.normalize_channels:
            transforms.append(Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]))

        # transform = Compose(transforms)
        preprocess = Compose(transforms)
    dataset = ImageFolder(
        path,
        transform=preprocess,
        target_transform=index_transform,
    )
    dataset.samples = [s_i for s_i in dataset.samples if s_i[1] in class_ids]
    return dataset

# get concatenated dataset for real (non-synthetic) data
def get_concat_dataset_real(
    args,
    is_train=False,
):
    """Concat datasamples in the correct order for each class."""
    if is_train:
        norms = args.train_norm
        class_ids = args.train_class_ids
        paths = args.train
    else:
        norms = args.val_norm
        class_ids = args.val_class_ids
        paths = args.val

    datasets = []
    for sub_dataset_idx, path in enumerate(paths):
        class_ids_sublist = (
            class_ids[sub_dataset_idx] if len(class_ids) > sub_dataset_idx else [0, 1]
        )
        norm = (
            norms[sub_dataset_idx] if norms and len(norms) > sub_dataset_idx else None
        )
        id_offset = len(
            [
                id
                for sublist_idx in range(sub_dataset_idx)
                for id in class_ids[sublist_idx]
            ]
        )  # how many class id's have been contained in the previous sub datasets
        dataset = get_dataset_real(
            is_train=is_train,
            args=args,
            path=path,
            class_ids=class_ids_sublist,
            norm=norm,
            id_offset=id_offset,
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)

# get training dataset splits for real (non-synthetic) data
def get_train_data_splits_real(args):
    """Split data if test set from train set is required."""
    training_dataset = get_concat_dataset_real(is_train=True, args=args)
    train_len = (
        args.n_train_samples if args.n_train_samples else len(training_dataset)
    ) - args.n_test_samples_from_train
    trash_len = (
        (len(training_dataset) - args.n_train_samples) if args.n_train_samples else 0
    )
    torch.manual_seed(torch.initial_seed())
    train_split, generated_train, trash = random_split(
        training_dataset, [train_len, args.n_test_samples_from_train, trash_len]
    )
    return train_split, generated_train

# get dataloader splits for real (non-synthetic) data
def get_dataloaders_real(args):
    """Construct the dataloader for the ImageNet (real) data."""
    train_split, generated_train = get_train_data_splits_real(args)
    # print(len(train_split))
    val_dataset = get_concat_dataset_real(args=args)

    # initialise dataloaders
    dataloader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,num_workers=args.num_workers)
    generated_train_dataloader = DataLoader(generated_train, batch_size=args.batch_size,num_workers=args.num_workers)

    return dataloader, val_dataloader, generated_train_dataloader
