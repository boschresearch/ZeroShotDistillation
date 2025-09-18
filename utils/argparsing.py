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

def add_train_args(parser):
# Parameters for the dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="Select dataset from: imagenet, pets, cars, food, flowers, aircraft, texture",
    )
    parser.add_argument(
        "--synthetic_data",
        type=str,
        default="False",
        help="Choose whether to use synthetic data or not",
    )
    parser.add_argument(
        "--train",
        type=str,
        help="The directory where the training data set is stored",
        action="append",
    )
    parser.add_argument(
        "--val",
        type=str,
        help="The directory where the validation data set is stored",
        action="append",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="The directory where the test data set is stored",
        action="append",
    )
    parser.add_argument(
        "--val_class_ids",
        nargs="+",
        action="append",
        type=int,
        help="Class ids of the classes to be used in the validation dataset. Will use the imagenet 100 ids in ascending order by default.",
    )
    parser.add_argument(
        "--train_class_ids",
        nargs="+",
        action="append",
        type=int,
        help="Class ids of the classes to be used in the training dataset. Will use ids 0-99 by default.",
    )
    parser.add_argument(
        "--val_int",
        type=int,
        default=8,
        help="Number of steps after which to perform a validation (repeated)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers that the dataloader uses. ",
    )
    parser.add_argument(
        "--n_test_samples_from_train",
        type=int,
        default=0,
        help="Number of samples to take from the train set as an optional set for imagenet - optional and not used.",
    )
# Important training and model hyperparameters
    parser.add_argument(
        "--model",
        type=str,
        default="tiny_vit_11m_224",
        help="Architecture of the student model, e.g. 'resnet18' or 'mobilenet_v2'. 'cct' to use a compact convolutional transformer.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="ViT-B-32",
        help="Architecture of the teacher model, select from open_clip",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs the model is trained for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use during model training",
        default=10,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate to use during model training",
        default=0.01,
    )
    parser.add_argument(
        "--log_temperature",
        type=float,
        default=0.0,
        help="Log temperature to use in the contrastive (distillation) loss.",
    )
    parser.add_argument(
        "--train_temperature",
        type=str,
        help="Set to True to use make the temperature a trainable parameter.",
        default="False",
    )
    parser.add_argument(
        "--distil_alpha",
        type=float_in_range(0.0, 1.0),
        default=0.5,
        help="Alpha used to balance the knowledge distillation loss against the cross-entropy loss derived from the ground truth. ",
    )
    parser.add_argument(
        "--contrastive_lambda",
        type=float_in_range(0.0, 1.0),
        default=0.5,
        help="Lambda used to balance the text-to-image and image-to-text loss. ",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Path to the location to save the model to after training. If this argument is not set, the model will not be saved.",
    )
    parser.add_argument(
        "--start_model_path",
        type=str,
        help="Path to the location to get the pretrained checkpoint to initialize the model. If this argument is not set, the model will not be initialized specifically.",
    )
    parser.add_argument(
        "--teacher_pretrained",
        type=str,
        default="openai",
        help="Pretraining dataset of teacher.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Dimension of the embedding space of the CLIP model.",
    )
    parser.add_argument(
        "--encoder_output_dim",
        type=int,
        default=512,
        help="Dimension of the output of the student vision encoder before the projection head.",
    )
    parser.add_argument(
        "--training_loss",
        type=str,
        help="Select supervised training loss from: contrastive, ce (cross-entropy)",
        required=True,
    )
    parser.add_argument(
        "--distillation_loss",
        type=str,
        help="Select distillation loss from: L2, cos, spherical, contrastive",
        required=True,
    )
    parser.add_argument(
        "--n_train_images_per_class",
        type=int,
        default=1000,
        help="Number of images per class in the training set.",
    )
    parser.add_argument(
        "--diverse_prompts",
        type=str,
        help="Set to True to use diverse prompts for each image in the training set.",
        default="False",
    )
    parser.add_argument(
        "--options_per_attribute",
        type=int,
        default=15,
        help="Number of options per attribute for the diverse images.",
    )
    parser.add_argument(
        "--use_diverse_prompts_for_inference",
        type=str,
        help="Set to True to use diverse prompts for inference.",
        default="False",
    )
    # Additional training and model hyperparameters
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Set this flag to train on CPU.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="The optimizer to train the model with. Expects the class name of the optimizer in torch.optim as string, e.g., 'AdamW'.",
        default="AdamW",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup epoch with increasing the learning rate until the full lr is reached (added to --epochs).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum used by optimizer. currently only used by SGD",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )
# logging
    parser.add_argument(
        "--logdir",
        type=str,
        help="The directory the logs will be saved to.",
        required=True,
    )
    parser.add_argument(
        "--logname",
        type=str,
        default="",
        help="Name of the experiment, will be used as name of the resulting tensorboard.",
        required=True,
    )
    parser.add_argument(
        "--save_checkpoints",
        type=str,
        help="Set to True to save checkpoints after every epoch",
        default="True",
    )
    parser.add_argument(
        "--logversion",
        default=0,
        help="Number of the version of the experiment.",
    )
# testing
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Directory names for checkpoints to be evaluated",
        nargs='+', 
        default=[],
    )
# hardware
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of gpus per node.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Set this flag to use mixed precision for training.",
    )
    return parser
