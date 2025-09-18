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

import torch
import timm
import os
import sys
import random
import time

import numpy as np
import torch.multiprocessing as mp

from torchvision import models
from torch import nn
from torch.nn import functional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from numpy.linalg import norm
from torchmetrics import Accuracy
from nltk.corpus import wordnet as wn
from uuid import uuid4
from torch import autocast

# Class is adapted from: https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py (visited 24/11/2023)
class ImageEncoder(nn.Module):
    """
        Class for the image encoder.
    """
    def __init__(self,args):
        super().__init__()
        self.model = timm.create_model(args.model, False, num_classes=0, global_pool="avg")

    def forward(self, x):
        return self.model(x)

# Class is taken from: https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py (visited 24/11/2023)
class Projection(nn.Module):
    """
        Class for the single-layer projection head. Two-layer projection head with non-linearity is commented.
    """
    def __init__(self,args):
        super().__init__()
        self.projection = nn.Linear(args.encoder_output_dim, args.embedding_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        return projected
