import argparse
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities import grad_norm
import torch
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from utils.CustomImageFolder import ImageCaptionFolder
from utils.StudentArchitectureComponents import ImageEncoder, Projection
from utils.datahandling_imagenet100 import (
    get_dataloaders, 
    get_dataloaders_real,
    get_train_data_splits_real, 
    get_train_data_splits,
    get_concat_dataset
)
from utils.datahandling_domainspecific import train_and_test_dataloader, train_dataloader_other, test_dataloader_other, get_test_data
from utils.datahandling_domainagnostic import (
    get_wds_loader,
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    CenterCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToPILImage,
)
from torchvision import models
from torchvision import datasets
from torch import nn
import sys
from torch.nn import functional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import clip
import open_clip
from nltk.corpus import wordnet as wn
import os
import sys
from torchmetrics.regression import KLDivergence
from utils.generate_prompts import inference_prompts, get_all_pairs_length,read_captions

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../../")
sys.path.insert(1, path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(parser):
    parser = add_train_args(parser)
    args = parser.parse_args()
    return post_parse(args)

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
# Important training and model hyperparameters
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        help="Architecture of the student model, e.g. 'resnet18' or 'mobilenet_v2'. 'cct' to use a compact convolutional transformer.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="ViT-B/32",
        help="Architecture of the teacher model, options are 'RN101' or 'ViT-B/32'",
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
        help="Lmabda used to balance the text-to-image and image-to-text loss. ",
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
        default=0.2,
        help="Weight decay",
    )
    parser.add_argument(
        "--batchnorm_train",
        action="store_true",
        default=False,
        help="Set this flag to set the batch norm layers to train mode during the last evaluation epoch.",
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


"""
    Use post_parse function for prompt and image generation
"""
def post_parse(args):
    if args.dataset=="imagenet":
        if args.train_class_ids[0][0]==1000:
            args.train_class_ids=[[i for i in range(1000)]]
        if args.val_class_ids!=None:
            if args.val_class_ids[0][0]==1000:
                args.val_class_ids=[[i for i in range(1000)]]
        train_class_ids = [id for sublist in args.train_class_ids for id in sublist]
    elif args.dataset=="pets":
        train_class_ids = [id for id in range(37)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    elif args.dataset=="food":
        train_class_ids = [id for id in range(101)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    elif args.dataset=="cars":
        train_class_ids = [id for id in range(196)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    elif args.dataset=="flowers":
        train_class_ids = [id for id in range(102)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    elif args.dataset=="texture":
        train_class_ids = [id for id in range(47)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    elif args.dataset=="aircraft":
        train_class_ids = [id for id in range(100)]
        args.train_class_ids = [train_class_ids]
        args.val_class_ids = [train_class_ids]
    if args.dataset!="datacomp":
        args.num_train_classes = len([id for sublist in args.train_class_ids for id in sublist])
    if args.train!=None:
        if eval(args.synthetic_data)==True:
            if eval(args.diverse_prompts):
                args.n_train_images_per_class=get_all_pairs_length(args.options_per_attribute)
            args.captions = read_captions(train_class_ids,args.n_train_images_per_class,args.train)
    return args

"""
    Utility functions for args
"""
def float_in_range(min, max):
    def verify_float_in_range(arg):
        try:
            value = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Argument must of type float")
        if value < min or value > max:
            raise argparse.ArgumentTypeError(
                f"Argument hast to be in range [{min}..{max}]"
            )
        return value

    return verify_float_in_range

def string_with_spaces(string):
    return str(string).replace("_", " ")

"""
    Contrastive loss
"""

def contrastive_loss(tokens1, tokens2, log_temperature):
    """loss for contrastive training."""
    # similarity matrix
    token_similarity = (tokens1 @ tokens2.T) * torch.exp(-log_temperature)
    # columnwise reduction of similarity matrix (i.e. over tokens2)
    dividend=torch.sum(torch.exp(token_similarity),dim=-1)
    # diagonal = similariy between same entries in tokens1 and token2
    contrastive_loss_vector=torch.exp(torch.diagonal(token_similarity, 0)) / dividend
    contrastive_loss_vector = (-1)*torch.log(contrastive_loss_vector)
    contrastive_loss = contrastive_loss_vector.mean()
    return contrastive_loss

"""
    Student Model
"""
class StudentModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.backends.cuda.matmul.allow_tf32 = True
            # set parameter
            self.args = args
            # data related attributes
            if args.dataset!="datacomp":
                self.num_train_classes = args.num_train_classes
            self.synthetic_data = args.synthetic_data
            self.train_dir = args.train
            # losses
            if args.training_loss=="KL":
                self.kl_divergence=KLDivergence()
            # model related hyperparameters and architectural parameters
            self.distil_alpha = args.distil_alpha
            self.contrastive_loss_lambda = args.contrastive_lambda
            # training related attributes
            self.learning_rate = args.learning_rate
            self.optimizer = args.optimizer
            self.momentum = args.momentum
            self.weight_decay = args.weight_decay
            self.distil_alpha = args.distil_alpha
            # self.lr_scheduling = args.lr_scheduling
            self.warmup = args.warmup
            self.epochs = args.epochs
            self.n_train_images_per_class = int(args.n_train_images_per_class)
            self.logdir = args.logdir
            self.logversion = args.logversion
            self.logname = args.logname
            if args.dataset!="datacomp":
                self.n_train_samples=args.num_train_classes*args.n_train_images_per_class
            # Checkpoint path
            if eval(args.save_checkpoints):
                self.checkpoint_path = os.path.join(self.logdir,self.logname,"version_"+str(self.logversion),"checkpoints")
            # load teacher
            self.teacher_name=args.teacher
            teacher_model, _, self.preprocess = open_clip.create_model_and_transforms(args.teacher, pretrained=args.teacher_pretrained)
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.teacher_model = teacher_model
        # Setting teacher parameters as fixed/non-trainable
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            #print(name)
        self.teacher_model.eval()
        # trainable parameter
        self.image_encoder = ImageEncoder(args)
        self.image_token_projector = Projection(args)
        if eval(args.train_temperature):
            self.temperature = nn.Parameter(data=args.log_temperature * torch.ones(1), requires_grad=True)
        else:
            self.temperature = nn.Parameter(data=args.log_temperature * torch.ones(1), requires_grad=False)
       
    def forward(self, x):
        image_encoded=self.image_encoder(x)
        return self.image_token_projector(image_encoded)

    def pass_image_encoder(self, x):
        image_encoded=self.image_encoder(x)
        return image_encoded


    def training_step(self, batch, batch_idx):
        # Get training batch
        x, y = batch
        # Format y correctly
        y = y.repeat(len(x)) if isinstance(x, list) else y
        # Compute text embeddings without influence on the gradient
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.teacher_model.eval()
            image_features_teacher = self.teacher_model.encode_image(x)
            image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
            # Normalize tokens
            image_features_teacher = image_features_teacher_normalized
            # text embedding
            if args.dataset!="datacomp":
                image_class_ids = [id for sublist in args.train_class_ids for id in sublist]
                photo_prompts = [inference_prompts(args.dataset,image_class_ids[class_num]) for class_num in y]
                photo_prompts_tokenized = self.tokenizer(photo_prompts).to(device)
                text_features = self.teacher_model.encode_text(photo_prompts_tokenized)
                # Normalize tokens
                text_features_normalized = functional.normalize(text_features, p=2, dim=-1)
                text_features = text_features_normalized
                # Compute logits of the teacher 
                logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
            else:
                text_tokenized = self.tokenizer(y).to(device)
                text_features = self.teacher_model.encode_text(text_tokenized)
                text_features_normalized = functional.normalize(text_features, p=2, dim=-1)
                # if (args.normalize_tokens==True):
                text_features = text_features_normalized
            logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
        # Compute image embeddings of the student
        image_features_student = self(x)
        # Normlize student tokens
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_student = image_features_student_normalized
        logits_student = (text_features @ image_features_student.T) * torch.exp(-self.temperature)


        # Compute distillation loss
        if self.distil_alpha>0.0:
            if args.distillation_loss=="L2":
                difference_to_teacher = image_features_student-image_features_teacher
                L2_distance_to_teacher = torch.norm(difference_to_teacher, dim=-1)
                distill_loss = sum(L2_distance_to_teacher)/len(L2_distance_to_teacher)
                
            elif args.distillation_loss=="I2I":
                # Distillation loss from https://arxiv.org/pdf/1503.02531.pdf 
                I2I_loss = contrastive_loss(image_features_student,image_features_teacher,self.temperature)
                distill_loss = I2I_loss
            else:
                print("Select appropriate feature loss")
        # Compute overall loss
        if self.distil_alpha<1.0:
            contrastive_loss_batch_1 = contrastive_loss(image_features_student, text_features, self.temperature)
            contrastive_loss_batch_2 = contrastive_loss(text_features, image_features_student, self.temperature)
            contrastive_loss_batch = 1/2*(contrastive_loss_batch_1+contrastive_loss_batch_2)
            print(contrastive_loss_batch)
        
        if self.distil_alpha==1.0:
            overall_loss = distill_loss
        elif self.distil_alpha==0.0:
            overall_loss = contrastive_loss_batch
        else:
            overall_loss = ((self.distil_alpha) * distill_loss) + ((1- self.distil_alpha) * contrastive_loss_batch)
        print("loss in current step: ", overall_loss)
        return overall_loss

    def validation_step(self, batch, batch_idx):
        self.teacher_model.eval()
        self.eval()
        if args.batchnorm_train:
            for layer in self.model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.train()
        x, y = batch
        # Compute tokens as for training
        image_class_ids = [id for sublist in args.val_class_ids for id in sublist]
        photo_prompts = [inference_prompts(args.dataset,image_class_ids[class_num]) for class_num in y] 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        photo_prompts_tokenized = self.tokenizer(photo_prompts).to(device)
        text_features = self.teacher_model.encode_text(photo_prompts_tokenized)
        image_features_teacher = self.teacher_model.encode_image(x)
        image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
        text_features_normalized = functional.normalize(text_features, p=2, dim=-1)
        image_features_teacher = image_features_teacher_normalized
        text_features = text_features_normalized
        logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
        reverse_logits_teacher = (image_features_teacher @ text_features.T) * torch.exp(self.temperature)
        image_features_student = self(x)
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_student = image_features_student_normalized
        #Compute loss as in training
        logits_student = (text_features @ image_features_student.T) * torch.exp(self.temperature)
        reverse_logits_student = (image_features_student @ text_features.T) * torch.exp(self.temperature)
        # Compute loss
        if self.distil_alpha>0.0:
            if args.distillation_loss=="L2":
                difference_to_teacher = image_features_student-image_features_teacher
                L2_distance_to_teacher = torch.norm(difference_to_teacher, dim=-1)
                distill_loss = sum(L2_distance_to_teacher)/len(L2_distance_to_teacher)
            elif args.distillation_loss=="I2I":
                # Distillation loss from https://arxiv.org/pdf/1503.02531.pdf 
                I2I_loss = contrastive_loss(image_features_student,image_features_teacher,self.temperature)
                distill_loss = I2I_loss
            else:
                print("Select appropriate feature loss")
        # Compute overall loss
        if self.distil_alpha<1.0:
            contrastive_loss_batch_1 = contrastive_loss(image_features_student, text_features, self.temperature)
            contrastive_loss_batch_2 = contrastive_loss(text_features, image_features_student, self.temperature)
            contrastive_loss_batch = 1/2*(contrastive_loss_batch_1+contrastive_loss_batch_2)
        if self.distil_alpha==1.0:
            overall_loss = distill_loss
        elif self.distil_alpha==0.0:
            overall_loss = contrastive_loss_batch
        else:
            overall_loss = ((self.distil_alpha) * distill_loss) + ((1- self.distil_alpha) * contrastive_loss_batch)
        # normalize tokens if not done before
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
        # Compute predictions if not done befor for CE loss
        photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in image_class_ids]
        if args.teacher!="ViT-B/32-DataCompXL" and args.teacher!="RN101_openclip" and args.teacher!=args.teacher=="ViT-B/16-LAION2B":  
            photo_prompts_per_class_tokenized = clip.tokenize(photo_prompts_per_class).to(device)
        else:
            photo_prompts_per_class_tokenized = self.tokenizer(photo_prompts_per_class).to(device)
        text_features_per_class = self.teacher_model.encode_text(photo_prompts_per_class_tokenized)
        text_features_per_class_normalized = functional.normalize(text_features_per_class, p=2, dim=-1)
        y_hat = (100.0 * image_features_student_normalized @ text_features_per_class_normalized.T).softmax(dim=-1)
        y_hat_teacher = (100.0 * image_features_teacher_normalized @ text_features_per_class_normalized.T).softmax(dim=-1)
        # print(teacher_predictions)log
        accuracy_student = torch.tensor(((torch.max(y_hat.data, 1)[1] == y).sum().item()) / y.size(0))
        accuracy_teacher = torch.tensor(((torch.max(y_hat_teacher.data, 1)[1] == y).sum().item()) / y.size(0))
        top5_accuracy_student = self.top5acc(y_hat, y)
        top5_accuracy_teacher = self.top5acc(y_hat_teacher, y)
        # log losses
        if self.distil_alpha<1.0:
            self.log('val_distillation_loss', distill_loss.item(), on_epoch=True, sync_dist=True)
        if self.distil_alpha>0.0:
            self.log('val_training_loss', contrastive_loss.item(), sync_dist=True)
        self.log('val_overall_loss', overall_loss.item(), sync_dist=True)
        self.log('val_accuracy_student', accuracy_student.item(), sync_dist=True)
        self.log('val_accuracy_teacher', accuracy_teacher.item(), sync_dist=True)
        self.log('val_top5_accuracy_student', top5_accuracy_student, sync_dist=True)
        self.log('val_top5_accuracy_teacher', top5_accuracy_teacher, sync_dist=True)
        print("validating")
        return y_hat, y


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if args.dataset == "datacomp" or args.dataset == "laion":
            train_dataloader = get_wds_loader(batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            train_dataloader = train_dataloader_other(self.args)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if args.dataset=="imagenet":
            if eval(args.synthetic_data)==True:
                train_split, generated_train = get_train_data_splits(self.args)
            else:
                train_split, generated_train = get_train_data_splits_real(self.args)

            val_dataset = get_concat_dataset(args=self.args, is_train=False)

            val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size,num_workers=args.num_workers,)
        else:
            val_dataloader = test_dataloader_other(self.args)
        return val_dataloader

    def test_step(self, batch, batch_idx):
        self.teacher_model.eval()
        self.eval()
        if args.batchnorm_train:
            for layer in self.model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.train()
        x, y = batch
        # Compute tokens as for training
        if args.dataset=="pets":
            complete_class_ids = range(37)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="food":
            complete_class_ids = range(101)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="flowers":
            complete_class_ids = range(102)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="texture":
            complete_class_ids = range(47)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="aircraft":
            complete_class_ids = range(100)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="cars":
            complete_class_ids = range(196)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        elif args.dataset=="imagenet":
            complete_class_ids = range(1000)
            photo_prompts = [inference_prompts(args.dataset,class_num.item()) for class_num in y]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.teacher!="ViT-B/32-DataCompXL" and args.teacher!="RN101_openclip" and args.teacher!=args.teacher=="ViT-B/16-LAION2B":
            print("Using CLIP tokenizer")
            photo_prompts_tokenized = clip.tokenize(photo_prompts).to(device)
        else:
            photo_prompts_tokenized = self.tokenizer(photo_prompts).to(device)
        text_features = self.teacher_model.encode_text(photo_prompts_tokenized)
        image_features_teacher = self.teacher_model.encode_image(x)
        image_features_teacher_normalized=image_features_teacher
        image_features_teacher_normalized /= image_features_teacher_normalized.norm(dim=-1, keepdim=True)
        text_features_normalized=text_features
        text_features_normalized /= text_features_normalized.norm(dim=-1, keepdim=True)
        image_features_teacher = image_features_teacher_normalized
        text_features = text_features_normalized
        logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
        reverse_logits_teacher = (image_features_teacher @ text_features.T) * torch.exp(-self.temperature)
        image_features_student = self(x)
        image_features_student_normalized=image_features_student
        image_features_student_normalized /= image_features_student_normalized.norm(dim=-1, keepdim=True)
        image_features_student = image_features_student_normalized
        #Compute loss as in training
        logits_student = (text_features @ image_features_student.T) * torch.exp(-self.temperature)
        reverse_logits_student = (image_features_student @ text_features.T) * torch.exp(-self.temperature)

        if self.distil_alpha>0.0:
            if args.distillation_loss=="L2":
                difference_to_teacher = image_features_student-image_features_teacher
                L2_distance_to_teacher = torch.norm(difference_to_teacher, dim=-1)
                distill_loss = sum(L2_distance_to_teacher)/len(L2_distance_to_teacher)
            elif args.distillation_loss=="I2I":
                # Distillation loss from https://arxiv.org/pdf/1503.02531.pdf 
                I2I_loss = contrastive_loss(image_features_student,image_features_teacher,self.temperature)
                distill_loss = I2I_loss
            else:
                print("Select appropriate feature loss")
        # Compute overall loss
        if self.distil_alpha<1.0:
            contrastive_loss_batch_1 = contrastive_loss(image_features_student, text_features, self.temperature)
            contrastive_loss_batch_2 = contrastive_loss(text_features, image_features_student, self.temperature)
            contrastive_loss_batch = 1/2*(contrastive_loss_batch_1+contrastive_loss_batch_2)
        if self.distil_alpha==1.0:
            overall_loss = distill_loss
        elif self.distil_alpha==0.0:
            overall_loss = contrastive_loss_batch
        else:
            overall_loss = ((self.distil_alpha) * distill_loss) + ((1- self.distil_alpha) * contrastive_loss_batch)
        # normalize tokens if not done before
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
        # Compute predictions
        if args.dataset=="pets":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="food":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="flowers":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="aircraft":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="texture":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="cars":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        elif args.dataset=="imagenet":
            photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        if args.teacher!="ViT-B/32-DataCompXL" and args.teacher!="RN101_openclip" and args.teacher!=args.teacher=="ViT-B/16-LAION2B":  
            photo_prompts_per_class_tokenized = clip.tokenize(photo_prompts_per_class).to(device)
        else:
            photo_prompts_per_class_tokenized = self.tokenizer(photo_prompts_per_class).to(device)
        text_features_per_class = self.teacher_model.encode_text(photo_prompts_per_class_tokenized)
        text_features_per_class /= text_features_per_class.norm(dim=-1, keepdim=True)
        y_hat = (100.0 * image_features_student_normalized @ text_features_per_class.T).softmax(dim=-1)
        y_hat_teacher = (100.0 * image_features_teacher_normalized @ text_features_per_class.T).softmax(dim=-1)
        accuracy_student = torch.tensor(((torch.max(y_hat.data, 1)[1] == y).sum().item()) / y.size(0))
        accuracy_teacher = torch.tensor(((torch.max(y_hat_teacher.data, 1)[1] == y).sum().item()) / y.size(0))
        top5_accuracy_student = self.top5acc(y_hat, y)
        top5_accuracy_teacher = self.top5acc(y_hat_teacher, y)
        top1_accuracy_student = self.top1acc(y_hat, y)
        top1_accuracy_teacher = self.top1acc(y_hat_teacher, y)

        # log losses and accuracy
        if self.distil_alpha>0.0:
            self.log("test_training_loss", contrastive_loss, on_epoch=True, on_step=True, sync_dist=True)
        if self.distil_alpha<1.0:
            self.log("test_distillation_loss", distill_loss, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_overall_loss", overall_loss, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_student_accuracy", top1_accuracy_student, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_teacher_accuracy", top1_accuracy_teacher, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_student_accuracy_top5", top5_accuracy_student, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_teacher_accuracy_top5", top5_accuracy_teacher, on_epoch=True, on_step=True, sync_dist=True)
        return y_hat, y

    
    
    def on_before_zero_grad(self, *args, **kwargs):
        # scale temperature to be between 1/100 and 100
        torch.clamp(self.temperature, min=-4.60517018599, max=4.60517018599)


    def get_parameter_groups_for_adamW(self):
        regularized_parameters = []
        non_regularized_parameters = []
        for parameter_name, parameter in self.image_encoder.named_parameters():
            #print(parameter_name)
            if not parameter.requires_grad:
                continue
            if len(parameter.shape) == 1 or parameter_name.endswith(".bias"):
                non_regularized_parameters.append(parameter)
            else:
                regularized_parameters.append(parameter)
        for parameter_name, parameter in self.image_token_projector.named_parameters():
            if not parameter.requires_grad:
                continue
            if len(parameter.shape) == 1 or parameter_name.endswith(".bias"):
                non_regularized_parameters.append(parameter)
            else:
                regularized_parameters.append(parameter)
        # add temperature to non-tegularized parameters
        non_regularized_parameters.append(self.temperature)
        return [
            {"params": regularized_parameters, "weight_decay": self.weight_decay},
            {"params": non_regularized_parameters, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.get_parameter_groups_for_sgd(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "AdamW":
            if self.teacher_name == "RN101":
                optimizer = getattr(torch.optim, self.optimizer)(
                    self.get_parameter_groups_for_adamW(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay
                )
            elif self.teacher_name == "ViT-L/14":
                optimizer = getattr(torch.optim, self.optimizer)(
                    self.get_parameter_groups_for_adamW(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-06, weight_decay=self.weight_decay
                )
            else: 
                optimizer = getattr(torch.optim, self.optimizer)(
                    self.get_parameter_groups_for_adamW(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-06, weight_decay=self.weight_decay
                )
        else:
            optimizer = getattr(torch.optim, self.optimizer)(
                self.parameters(), lr=self.learning_rate
            )

        if args.warmup>0:
            steps_per_epoch = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches #len(self.trainer.train_dataloader)
            non_warmup_epochs = self.epochs
            non_warmup_steps = non_warmup_epochs * steps_per_epoch
            warmup_epochs = int(self.warmup)
            warmup_steps = warmup_epochs * steps_per_epoch
            lr_schedulers = [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=1 / warmup_steps,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer=optimizer,
                    total_iters=non_warmup_steps,
                    factor=1.0,
                ),

            ]
            lr_scheduler_with_warmup = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=lr_schedulers,
                milestones=[warmup_steps],
            )

            lr_scheduler_config = {
                "scheduler": lr_scheduler_with_warmup,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler_config]     
        else: 
            return optimizer

"""
   Main
"""

def main(args):
    # # set up logging
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.logname)
    log_int = 8
    if args.logversion==None:
        args.logversion=logger.version

    # # initialize student model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentModel(args).to(device)
    # Instantiate model from pre-trained checkpoint
    if args.start_model_path:
        print("Initializing model")
        checkpoint = torch.load(args.start_model_path)
        model.load_state_dict(checkpoint["state_dict"],strict=False)
    # Configure checkpointing after every epoch
    if eval(args.save_checkpoints):
        checkpoint_callback = ModelCheckpoint(
            filename="fa_classifier_{epoch:02d}",
            every_n_epochs=log_int,
            save_top_k=-1,  
        )
            
    print("Number of GPUs:", torch.cuda.device_count())
    # train student model
    if args.val!=None:
        # Configure checkpointing after every epoch
        checkpoint_callback = ModelCheckpoint(
            filename="fa_classifier_{epoch:02d}",
            every_n_epochs=log_int,
            save_top_k=-1,  
        )
        if torch.cuda.device_count() > 1 and not args.cpu:
            print("Number of GPUs available: ", torch.cuda.device_count())
            trainer = Trainer(
                accelerator="cuda",
                enable_checkpointing=eval(args.save_checkpoints),
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                devices=args.devices,
                num_nodes=args.nodes,
                sync_batchnorm=True,
                precision=16 if args.mixed_precision else 32,
                max_epochs=args.epochs+args.warmup,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                callbacks=[checkpoint_callback],
                strategy=DDPStrategy(find_unused_parameters=True),
            )
        elif torch.cuda.device_count() == 1 and not args.cpu:
            print("Training on one gpu")
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            trainer = Trainer(
                accelerator="cuda",
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                max_epochs=args.epochs+args.warmup,
                enable_checkpointing=eval(args.save_checkpoints),
                precision=16 if args.mixed_precision else 32,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                reload_dataloaders_every_n_epochs=1,
                callbacks=[lr_monitor,checkpoint_callback],
            )
        else:
            print("Training cpu")
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=args.epochs+args.warmup,
                enable_checkpointing=eval(args.save_checkpoints),
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                precision=16 if args.mixed_precision else 32,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                callbacks=[checkpoint_callback], 
            )
    else:
        print("training without validation")
        if args.dataset=="datacomp":
            checkpoint_callback = ModelCheckpoint(
                filename="checkpoint_step_{step}",
                every_n_train_steps=1000,
                save_top_k=-1,  
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                filename="checkpoint_{epoch:02d}",
                every_n_epochs=log_int,
                save_top_k=-1,  
            )
        if torch.cuda.device_count() > 1 and not args.cpu:
            trainer = Trainer(
                accelerator="cuda",
                enable_checkpointing=eval(args.save_checkpoints),
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                devices=args.devices,
                num_nodes=args.nodes,
                sync_batchnorm=True,
                precision=16 if args.mixed_precision else 32,
                max_epochs=args.epochs+args.warmup,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                callbacks=[checkpoint_callback],
                limit_val_batches = 0.0,
                num_sanity_val_steps=0,
                strategy=DDPStrategy(find_unused_parameters=True),
            )
        elif torch.cuda.device_count() == 1 and not args.cpu:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer = Trainer(
                accelerator="cuda",
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                max_epochs=args.epochs+args.warmup,
                enable_checkpointing=eval(args.save_checkpoints),
                precision=16 if args.mixed_precision else 32,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                callbacks=[lr_monitor,checkpoint_callback],
                limit_val_batches = 0.0,
                num_sanity_val_steps=0,
            )
        else:
            trainer = Trainer(
                accelerator="cpu",
                max_epochs=args.epochs+args.warmup,
                enable_checkpointing=eval(args.save_checkpoints),
                default_root_dir=os.path.join(args.logdir,args.logname,"version_"+str(args.logversion)),
                precision=16 if args.mixed_precision else 32,
                enable_progress_bar=True,
                logger=logger,
                log_every_n_steps=log_int,
                callbacks=[checkpoint_callback], 
                limit_val_batches = 0.0,
                num_sanity_val_steps=0,
            )
    if args.val!=None:
        trainer.validate(model,)
    # perform training
    if args.train!=None:
        trainer.fit(
            model,
        )
    # save model
    if args.save_model_path:
        trainer.save_checkpoint(args.save_model_path)

    # test model
    if args.test!=None:
        if eval(args.synthetic_data):
            in_dir = args.test[0]
            out_dir = os.path.join(in_dir, "images")
            test_transform = Compose(
                        [
                            Resize(224),
                            CenterCrop(224),
                            ToTensor(),
                            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ]
                    )
            test_data=ImageFolder(out_dir,transform=test_transform)
        elif args.dataset=="imagenet":
            args.train=args.test
            args.val=args.test
            if eval(args.synthetic_data)==True:
                not_needed_dataloader1, test_dataloader, not_needed_dataloader2 = get_dataloaders(
                    args=args
                )
            else:
                args.n_test_samples=len(args.val_class_ids[0])*2000#args.n_test_images_per_class
                args.n_train_samples=args.n_test_samples
                args.n_val_samples=args.n_test_samples
                not_needed_dataloader1, test_dataloader, not_needed_dataloader1 = get_dataloaders_real(
                    args=args
                )
        else:
            print("Using normalization for CLIP models")
            test_transform = Compose(
                    [
                        Resize(224),
                        CenterCrop(224),
                        ToTensor(),
                        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                    ]
                )
            if args.dataset=="pets":
                test_data = datasets.OxfordIIITPet(root=args.test[0],split="test",transform=test_transform)
                train_data = datasets.OxfordIIITPet(root=args.test[0],split="trainval",transform=test_transform)
            elif args.dataset=="flowers":
                test_data = datasets.Flowers102(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="texture":
                test_data = datasets.DTD(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="aircraft":
                test_path = os.path.join(args.test[0],"test")
                test_data = datasets.ImageFolder(root=test_path,transform=test_transform)
                train_path = os.path.join(args.test[0],"train")
            elif args.dataset=="cars":
                test_data = datasets.StanfordCars(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="food":
                test_data = datasets.Food101(root=args.test[0],split="test",transform=test_transform)
        if args.dataset!="imagenet":
            test_dataloader=DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,drop_last=False)
        distillation_losses = []
        training_losses = []
        overall_losses = []
        accuracy_teacher = []
        accuracy_student = []
        top5_accuracy_teacher = []
        top5_accuracy_student = []
        for checkpoint in args.checkpoints:
            print("Loading model from ", checkpoint)
            checkpoint_loaded = torch.load(checkpoint)
            model.load_state_dict(checkpoint_loaded["state_dict"])
            model.freeze()
            result_dict = trainer.test(model,dataloaders=test_dataloader)
            distillation_losses.append(result_dict[0]["test_distillation_loss_epoch"])
            training_losses.append(result_dict[0]["test_training_loss_epoch"])
            overall_losses.append(result_dict[0]["test_overall_loss_epoch"])
            accuracy_student.append(result_dict[0]["test_student_accuracy_epoch"])
            accuracy_teacher.append(result_dict[0]["test_teacher_accuracy_epoch"])
            top5_accuracy_student.append(result_dict[0]["test_student_accuracy_top5_epoch"])
            top5_accuracy_teacher.append(result_dict[0]["test_teacher_accuracy_top5_epoch"])
        print("Distillation losses", distillation_losses)
        print("Training losses", training_losses)
        print("Overall losses", overall_losses)
        print("Accuracy teacher", accuracy_teacher)
        print("Accuracy student", accuracy_student)
        print("Top5 Accuracy teacher", top5_accuracy_teacher)
        print("Top5 Accuracy student", top5_accuracy_student)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Configuration")
    args = parse_args(parser)
    main(args)
