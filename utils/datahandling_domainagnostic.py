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
# -*- coding: utf-8 -*

import argparse
import torch
import os
import pytest
import math

import webdataset as wds
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from torch.nn import functional
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from typing import Callable
from torch.utils.data import DataLoader
from multiprocessing import Value
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    CenterCrop,
)
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

LAION_FILES = "/path/to/LAION/data"

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_transform() -> Compose:
    """Return ImageNet transformation."""
    return Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            # Choose between ImageNet and CLIP normalization, both work well: https://github.com/openai/CLIP/issues/20
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
    )

def filter_missing_data(sample: dict) -> bool:
    """Filter out samples with missing information."""
    has_caption = "txt" in sample
    image_extensions = {"png", "jpg", "jpeg", "webp"}
    has_image = any(extension in sample for extension in image_extensions)
    has_json = "json" in sample
    return has_caption and has_image and has_json


def get_wds_loader(batch_size: int = 256, num_workers: int = 4) -> DataLoader:
    """Create and return DataLoader over the DataComp dataset."""
    print("=> using WebDataset loader for single gpu.")
    transform = get_transform()
    dataset = (
        wds.WebDataset(LAION_FILES)  # dict with keys: jpg, json, txt, etc.
        .decode("pil")  # decode data
        .rename(image="jpg;png;jpeg;webp", json="json", caption="txt")
        .to_tuple("image", "caption")  # convert to (image, caption)
        .map_tuple(transform, lambda x: x)  # transform image
    )
    return DataLoader(
        dataset.batched(batch_size),
        num_workers=num_workers,
        batch_size=None,
    )
       
