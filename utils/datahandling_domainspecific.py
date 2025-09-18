#!/usr/local/bin/python3
# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# -*- coding: utf-8 -*-


import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import scipy
import os

from torchvision.datasets import ImageFolder, ImageNet
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    CenterCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from torchvision import datasets
from torch import nn
from torch.nn import functional

from utils.CustomImageFolder import ImageCaptionFolder


def train_and_test_dataloader(args):
    """Return train dataloader and dummy validataion/test dataloader for domainspecific datasets."""
    ImageNet_transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )
    if args.dataset=="pets":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.OxfordIIITPet(
                root=args.train[0],
                split="trainval",
                transform=ImageNet_transform
            )
        test_data = datasets.OxfordIIITPet(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="food":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.Food101(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
        test_data = datasets.Food101(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="cars":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.StanfordCars(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
        test_data = datasets.StanfordCars(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="flowers":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.Flowers102(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
        test_data = datasets.Flowers102(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="aircraft":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_path = os.path.join(args.train[0],"train")
            train_data = datasets.ImageFolder(root=train_path,transform=ImageNet_transform)
        test_path = os.path.join(args.val[0],"test")
        test_data = datasets.ImageFolder(root=test_path,transform=ImageNet_transform)
    elif args.dataset=="texture":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.DTD(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
        test_data = datasets.DTD(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    
    train_dataloader=DataLoader(train_data, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=True,
                                # drop_last=True,
                                )
    test_dataloader=DataLoader(test_data, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,)
    return train_dataloader, test_dataloader

def test_dataloader_other(args):
    """Return validataion/test dataloader for domainspecific datasets."""
    ImageNet_transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )
    if args.dataset=="pets":
        test_data = datasets.OxfordIIITPet(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="food":
        test_data = datasets.Food101(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="flowers":
        test_data = datasets.Flowers102(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="aircraft":
        test_path = os.path.join(args.val[0],"test")
        test_data = datasets.ImageFolder(root=test_path,transform=ImageNet_transform)
    elif args.dataset=="texture":
        test_data = datasets.DTD(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="cars":
        test_data = datasets.StanfordCars(
            root=args.val[0],
            split="test",
            transform=ImageNet_transform
        )
    test_dataloader=DataLoader(test_data, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,)
    return test_dataloader

def train_dataloader_other(args):
    """Return train dataloader for domainspecific datasets."""
    ImageNet_transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )
    if args.dataset=="pets":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.OxfordIIITPet(
                root=args.train[0],
                split="trainval",
                transform=ImageNet_transform
            )
    elif args.dataset=="food":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.Food101(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
    elif args.dataset=="flowers":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.Flowers102(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
    elif args.dataset=="aircraft":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_path = os.path.join(args.train[0],"train")
            train_data = datasets.ImageFolder(root=train_path,transform=ImageNet_transform)
    elif args.dataset=="texture":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.DTD(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
    elif args.dataset=="cars":
        if eval(args.synthetic_data):
            if eval(args.use_diverse_prompts_for_inference):
                train_data = ImageCaptionFolder(args.train[0],transform=ImageNet_transform,)
            else:
                train_dir = os.path.join(args.train[0], "images")
                train_data = ImageFolder(train_dir,transform=ImageNet_transform,)
        else:
            train_data = datasets.StanfordCars(
                root=args.train[0],
                split="train",
                transform=ImageNet_transform
            )
    train_dataloader=DataLoader(train_data, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                shuffle=True,
                                )
    return train_dataloader
