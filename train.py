import argparse
import sys
import torch
# import clip # removed openai CLIP models since they are integrated into open_clip
import open_clip
import os
import sys

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
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
from torch.nn import functional
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from nltk.corpus import wordnet as wn
from torchmetrics.regression import KLDivergence
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy

from utils.generate_prompts import inference_prompts, get_all_pairs_length,read_captions
from utils.CustomImageFolder import ImageCaptionFolder
from utils.StudentArchitectureComponents import ImageEncoder, Projection
from utils.datahandling_domainspecific import train_and_test_dataloader, train_dataloader_other, test_dataloader_other, get_test_data
from utils.datahandling_domainagnostic import get_wds_loader
from utils.argparsing import add_train_args

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, "../../")
sys.path.insert(1, path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args(parser):
    parser = add_train_args(parser)
    args = parser.parse_args()
    return post_parse(args)

def post_parse(args):
    """
    Use post_parse function to initialize the class(numbers) for different dataset
    """
    if args.dataset=="pets":
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


def float_in_range(min, max):
    """
    Utility function to check if a float is in a range
    """
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
    """
    Utility function to replace _ in a string
    """
    return str(string).replace("_", " ")

def contrastive_loss(features1, features2, log_temperature):
    """loss for contrastive training."""
    # similarity matrix
    token_similarity = (features1 @ features2.T) * torch.exp(-log_temperature)
    # columnwise reduction of similarity matrix (i.e. over features2)
    dividend=torch.sum(torch.exp(token_similarity),dim=-1)
    # diagonal = similariy between same entries in features1 and token2
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
            self.top1acc = Accuracy(task="multiclass", num_classes=args.num_train_classes, top_k=1)
            self.synthetic_data = args.synthetic_data
            self.train_dir = args.train
            # model related hyperparameters and architectural parameters
            self.distil_alpha = args.distil_alpha
            self.contrastive_loss_lambda = args.contrastive_lambda
            # training related attributes
            self.learning_rate = args.learning_rate
            self.optimizer = args.optimizer
            self.momentum = args.momentum
            self.weight_decay = args.weight_decay
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
            self.tokenizer = open_clip.get_tokenizer(args.teacher)
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
        """
        Forward pass: encode image and project
        """
        image_encoded=self.image_encoder(x)
        return self.image_token_projector(image_encoded)

    def pass_image_encoder(self, x):
        """
        Encode image without projection
        """
        image_encoded=self.image_encoder(x)
        return image_encoded


    def training_step(self, batch, batch_idx):
        """
        Single training step
        """
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
            # Normalize features
            image_features_teacher = image_features_teacher_normalized
            # text embedding
            if args.dataset!="datacomp":
                image_class_ids = [id for sublist in args.train_class_ids for id in sublist]
                photo_prompts = [inference_prompts(args.dataset,image_class_ids[class_num]) for class_num in y]
                photo_prompts_tokenized = self.tokenizer(photo_prompts).to(device)
                text_features = self.teacher_model.encode_text(photo_prompts_tokenized)
                # Normalize features
                text_features_normalized = functional.normalize(text_features, p=2, dim=-1)
                text_features = text_features_normalized
                # Compute logits of the teacher 
                logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
            else:
                text_tokenized = self.tokenizer(y).to(device)
                text_features = self.teacher_model.encode_text(text_tokenized)
                text_features_normalized = functional.normalize(text_features, p=2, dim=-1)
                text_features = text_features_normalized
            logits_teacher = (text_features @ image_features_teacher.T) * torch.exp(-self.temperature)
        # Compute image embeddings of the student
        image_features_student = self(x)
        # Normlize student features
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
        
        if self.distil_alpha==1.0:
            overall_loss = distill_loss
        elif self.distil_alpha==0.0:
            overall_loss = contrastive_loss_batch
        else:
            overall_loss = ((self.distil_alpha) * distill_loss) + ((1- self.distil_alpha) * contrastive_loss_batch)
        print("loss in current step: ", overall_loss)
        return overall_loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step
        """
        self.teacher_model.eval()
        self.eval()
        x, y = batch
        # Compute features as for training
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
        reverse_logits_teacher = (image_features_teacher @ text_features.T) * torch.exp(-self.temperature)
        image_features_student = self(x)
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_student = image_features_student_normalized
        #Compute loss as in training
        logits_student = (text_features @ image_features_student.T) * torch.exp(-self.temperature)
        reverse_logits_student = (image_features_student @ text_features.T) * torch.exp(-self.temperature)
        # Compute loss
        if self.distil_alpha>0.0:
            if args.distillation_loss=="L2":
                difference_to_teacher = image_features_student-image_features_teacher
                L2_distance_to_teacher = torch.norm(difference_to_teacher, dim=-1)
                distill_loss = sum(L2_distance_to_teacher)/len(L2_distance_to_teacher) # take mean per batch
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
        # normalize features if not done before
        image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
        # Compute predictions if not done befor for CE loss
        photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in image_class_ids]
        photo_prompts_per_class_tokenized = self.tokenizer(photo_prompts_per_class).to(device)
        text_features_per_class = self.teacher_model.encode_text(photo_prompts_per_class_tokenized)
        text_features_per_class_normalized = functional.normalize(text_features_per_class, p=2, dim=-1)
        y_hat = (100.0 * image_features_student_normalized @ text_features_per_class_normalized.T).softmax(dim=-1)
        y_hat_teacher = (100.0 * image_features_teacher_normalized @ text_features_per_class_normalized.T).softmax(dim=-1)
        # print(teacher_predictions)log
        accuracy_student = torch.tensor(((torch.max(y_hat.data, 1)[1] == y).sum().item()) / y.size(0))
        accuracy_teacher = torch.tensor(((torch.max(y_hat_teacher.data, 1)[1] == y).sum().item()) / y.size(0))
        # log losses
        if self.distil_alpha>0.0:
            self.log('val_distillation_loss', distill_loss.item(), on_epoch=True, sync_dist=True)
        if self.distil_alpha<1.0:
            self.log('val_training_loss', contrastive_loss_batch.item(), sync_dist=True)
        self.log('val_overall_loss', overall_loss.item(), sync_dist=True)
        self.log('val_accuracy_student', accuracy_student.item(), sync_dist=True)
        self.log('val_accuracy_teacher', accuracy_teacher.item(), sync_dist=True)
        print("validated")
        return y_hat, y


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        set up train dataloader, select between domain-agnostic or domain-specific dataloader
        """
        if args.dataset == "datacomp" or args.dataset == "laion":
            train_dataloader = get_wds_loader(batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            train_dataloader = train_dataloader_other(self.args)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        set up validation loader
        """
        val_dataloader = test_dataloader_other(self.args)
        return val_dataloader

    def test_step(self, batch, batch_idx):
        """
        perform evaluation of a model checkpoint
        """
        self.teacher_model.eval()
        self.eval()
        x, y = batch
        # Compute features as for training
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # normalize features if not done before
        # image_features_student_normalized = functional.normalize(image_features_student, p=2, dim=-1)
        # image_features_teacher_normalized = functional.normalize(image_features_teacher, p=2, dim=-1)
        # Compute predictions
        photo_prompts_per_class = [inference_prompts(args.dataset,class_num) for class_num in complete_class_ids]
        photo_prompts_per_class_tokenized = self.tokenizer(photo_prompts_per_class).to(device)
        text_features_per_class = self.teacher_model.encode_text(photo_prompts_per_class_tokenized)
        text_features_per_class /= text_features_per_class.norm(dim=-1, keepdim=True)
        y_hat = (100.0 * image_features_student @ text_features_per_class.T).softmax(dim=-1)
        y_hat_teacher = (100.0 * image_features_teacher @ text_features_per_class.T).softmax(dim=-1)
        accuracy_student = torch.tensor(((torch.max(y_hat.data, 1)[1] == y).sum().item()) / y.size(0))
        accuracy_teacher = torch.tensor(((torch.max(y_hat_teacher.data, 1)[1] == y).sum().item()) / y.size(0))
        # top5_accuracy_student = self.top5acc(y_hat, y)
        # top5_accuracy_teacher = self.top5acc(y_hat_teacher, y)
        top1_accuracy_student = self.top1acc(y_hat, y)
        top1_accuracy_teacher = self.top1acc(y_hat_teacher, y)

        # log losses and accuracy
        if self.distil_alpha<1.0:
            self.log("test_training_loss", contrastive_loss_batch, on_epoch=True, on_step=True, sync_dist=True)
        if self.distil_alpha>0.0:
            self.log("test_distillation_loss", distill_loss, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_overall_loss", overall_loss, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_student_accuracy", top1_accuracy_student, on_epoch=True, on_step=True, sync_dist=True)
        self.log("test_teacher_accuracy", top1_accuracy_teacher, on_epoch=True, on_step=True, sync_dist=True)
        # self.log("test_student_accuracy_top5", top5_accuracy_student, on_epoch=True, on_step=True, sync_dist=True)
        # self.log("test_teacher_accuracy_top5", top5_accuracy_teacher, on_epoch=True, on_step=True, sync_dist=True)
        return y_hat, y

    
    
    def on_before_zero_grad(self, *args, **kwargs):
        """
        scale temperature to be between 1/100 and 100
        """
        torch.clamp(self.temperature, min=-4.60517018599, max=4.60517018599)


    def get_parameter_groups_for_adamW(self):
        """
        get regularized/non-regularized parameters for weight decay (if selected)
        """
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
        """
        Configure optimizer
        """
        if self.optimizer == "AdamW":
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
                            Resize(256),
                            CenterCrop(224),
                            ToTensor(),
                            #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ]
                    )
            test_data=ImageFolder(out_dir,transform=test_transform)
        else:
            test_transform = Compose(
                    [
                        Resize(256), # for the paper we used 224 here which cuts less from the background in the centercrop
                        CenterCrop(224),
                        ToTensor(),
                        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                    ]
                )
            if args.dataset=="pets":
                test_data = datasets.OxfordIIITPet(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="flowers":
                test_data = datasets.Flowers102(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="texture":
                test_data = datasets.DTD(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="aircraft":
                test_path = os.path.join(args.test[0],"test")
                test_data = datasets.ImageFolder(root=test_path,transform=test_transform)
            elif args.dataset=="cars":
                test_data = datasets.StanfordCars(root=args.test[0],split="test",transform=test_transform)
            elif args.dataset=="food":
                test_data = datasets.Food101(root=args.test[0],split="test",transform=test_transform)
        test_dataloader=DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,drop_last=False)
        distillation_losses = []
        training_losses = []
        overall_losses = []
        accuracy_teacher = []
        accuracy_student = []
        # top5_accuracy_teacher = []
        # top5_accuracy_student = []
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
            # top5_accuracy_student.append(result_dict[0]["test_student_accuracy_top5_epoch"])
            # top5_accuracy_teacher.append(result_dict[0]["test_teacher_accuracy_top5_epoch"])
        print("Distillation losses", distillation_losses)
        print("Training losses", training_losses)
        print("Overall losses", overall_losses)
        print("Accuracy teacher", accuracy_teacher)
        print("Accuracy student", accuracy_student)
        # print("Top5 Accuracy teacher", top5_accuracy_teacher)
        # print("Top5 Accuracy student", top5_accuracy_student)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Configuration")
    args = parse_args(parser)
    main(args)
