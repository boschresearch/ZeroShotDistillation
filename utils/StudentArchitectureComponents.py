"""
    Classes for the student architecture components.
"""
import torch
import timm
import os
import sys
import random
import time
import clip
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
