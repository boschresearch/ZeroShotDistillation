"""
    Functions for handling real and synthetic data for OxfordIIITPet, Food101
"""
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import scipy
import os

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from utils.CustomImageFolder import ImageCaptionFolder
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


def train_and_test_dataloader(args):
    print("getting data from utils")
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
    ImageNet_transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
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

def get_test_data(args):
    ImageNet_transform = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )
    if args.dataset=="pets":
        train_data = datasets.OxfordIIITPet(
            root=args.test[0],
            split="trainval",
            transform=ImageNet_transform
        )
        test_data = datasets.OxfordIIITPet(
            root=args.test[0],
            split="test",
            transform=ImageNet_transform
        )
    elif args.dataset=="food":
        test_data = datasets.Food101(
            root=args.test[0],
            split="test",
            transform=ImageNet_transform
        )
    else:
        print("No suitable test dataset selected")
