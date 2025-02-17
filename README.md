# Feature Distillation Improves Zero-Shot Transfer from Synthetic Images

[![TMLR Paper](https://img.shields.io/badge/TMLR-Paper-blue)](https://openreview.net/forum?id=SP8DLl6jgb)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=KbdacNWGiAM)


This is the code release for the TMLR publication "Feature Distillation Improves Zero-Shot Transfer from Synthetic Images" by Niclas Popp, Jan Hendrik Metzen and Matthias Hein. The following figure summarizes the experimental setup:


The codebase consists of three components: 

1. Domain-agnostic Distillation
2. Synthetic Data Generation
3. Domain-specific Distillation

The required packages are listed in the requirements.txt file. The code was tested on NVIDIA v100 and h100 gpus.

## Step 1: Domain-agnostic Distillation
For an example file to run domain-agnostic distillation together with the available hyperparameters see: [example_domain_agnostic.sh](https://github.com/boschresearch/ZeroShotDistillation/blob/main/example_domain_agnostic.sh)
The code is built to use the webdataset dataloader together with .tar files. For details in how to setup the data for this kind of dataloader see [here](https://github.com/webdataset/webdataset)

## Step 2: Synthetic data Generation
The synthetic data generation process can be started as shown in the [example_data_generation.sh](https://github.com/boschresearch/ZeroShotDistillation/blob/main/example_data_generation.sh) file. 
For different domains, select the corresponding dataset option.

## Step 3: Domain-specific Distillation
The final step of our framework is domain-specific distillation.
An example together with the available options is given in the file [example_domain_specific.sh](https://github.com/boschresearch/ZeroShotDistillation/blob/main/example_domain_specific.sh). 
This step requires the final model checkpoint from step 1 and the synthetic data from step 2.
