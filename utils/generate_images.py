"""
    Functions for generating images with and without CLIP filtering.
"""
import torch
import numpy as np
# import clip
import open_clip # switched to open_clip as the openai models are included there
import os

from utils.generate_prompts import get_dedicated_class_names
from diffusers import AutoPipelineForText2Image, LCMScheduler, DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
from accelerate import PartialState

# # SDXL lora
Image_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
Image_adapter_id = "latent-consistency/lcm-lora-sdxl"
Image_pipe = AutoPipelineForText2Image.from_pretrained(Image_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
Image_pipe.load_lora_weights(Image_adapter_id)
Image_pipe.fuse_lora()


def generate_images(args):
    """Generate images based on images caption stored in args."""
    # check/make data directory
    savedir = args.train[0]
    os.makedirs(savedir, exist_ok=True)
    image_path = os.path.join(savedir, "images")
    os.makedirs(image_path, exist_ok=True)
    # check/make subdirectories for every class
    imagenet_class_ids = [id for sublist in args.train_class_ids for id in sublist]
    for class_id in imagenet_class_ids:
        path = os.path.join(savedir, "images", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
    # load pipeline
    model_id = "Lykon/dreamshaper-7"
    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
    pipe.safety_checker = disabled_safety_checker
    pipe.to("cuda")
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    for index in range(len(imagenet_class_ids)):
        for index2 in range(args.n_train_images_per_class):
            # tic = time.perf_counter()
            image_caption=args.captions[index*args.n_train_images_per_class+index2]
            prompt_embeds = compel_proc(image_caption)
            image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=6, guidance_scale=args.guidance_scale,requires_safety_checker=False,safety_checker=None,).images[0]
            image_path = os.path.join(
                    savedir,
                    "images",
                    f"{imagenet_class_ids[index]:03d}",
                    f"{index2:03d}.png",
                )
            image.save(image_path)
            # toc = time.perf_counter()
            # print(f"Time for generating the image was {toc - tic:0.4f} seconds")

def generate_images_accelerate(args):
    """Generate images based on images caption stored in args."""
    # get rank
    state = PartialState()
    rank = state.process_index
    # check/make data directory
    savedir = args.train[0]
    os.makedirs(savedir, exist_ok=True)
    image_path = os.path.join(savedir, "images")
    os.makedirs(image_path, exist_ok=True)
    # check/make subdirectories for every class
    imagenet_class_ids = [id for sublist in args.train_class_ids for id in sublist]
    # split class ids between ranks
    imagenet_class_ids_local = np.array_split(imagenet_class_ids, args.devices)[rank]
    imagenet_class_ids_local_counter = np.array_split(np.array(range(len(imagenet_class_ids))), args.devices)[rank]
    # imagenet_class_ids_local = [imagenet_class_ids_local[-1]]
    print("local rank:", rank)
    print("local class ids:",imagenet_class_ids_local)
    for class_id in imagenet_class_ids_local:
        path = os.path.join(savedir, "images", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
    
    # load pipeline
    Image_pipe.to(state.device)
    def disabled_safety_checker(images, clip_input):
        if len(images.shape)==4:
            num_images = images.shape[0]
            return images, [False]*num_images
        else:
            return images, False
    Image_pipe.safety_checker = disabled_safety_checker

    # if using SDXL
    Image_pipe.scheduler = LCMScheduler.from_config(Image_pipe.scheduler.config)
    compel = Compel(
            tokenizer=[Image_pipe.tokenizer, Image_pipe.tokenizer_2] ,
            text_encoder=[Image_pipe.text_encoder, Image_pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
            )
    
     # add CLIP filter
    if eval(args.CLIP_filter):
        print("filter images by teacher")
        model, _, preprocess = open_clip.create_model_and_transforms(args.teacher, pretrained=args.teacher_pretrained)
        tokenizer = open_clip.get_tokenizer(args.teacher)
        class_names=get_dedicated_class_names(args.dataset,args.train_class_ids[0])
        text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in class_names]).to(rank)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for index in range(len(imagenet_class_ids_local)):
        # for index2 in range(XX,args.n_train_images_per_class):
        index2=0
        while index2<args.n_train_images_per_class:
            # tic = time.perf_counter()
            print("local rank:", rank)
            print("Generating (accelerated) image: ", index2, "for class: ", imagenet_class_ids_local[index])
            # remember to shift entry by first class for current rank
            image_caption=args.captions[(imagenet_class_ids_local_counter[0]+index)*args.n_train_images_per_class+index2]
          
            # if using SDXL
            print(image_caption)
            conditioning, pooled = compel(image_caption)
            image = Image_pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, num_inference_steps=6, guidance_scale=args.guidance_scale,requires_safety_checker=False,safety_checker=None,).images[0]

 
            if eval(args.CLIP_filter):
                # CLIP filter
                image_input = preprocess(image).unsqueeze(0).to(state.device)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)
                # by creating the text features the classes are renumbered from 0 to N-1
                if (indices[0]==int(imagenet_class_ids_local[index])):
                    print("Image accepted")
                    image_path = os.path.join(
                        savedir,
                        "images",
                        f"{imagenet_class_ids_local[index]:03d}",
                        f"{index2:03d}.png",
                    )
                    image.save(image_path)
                    index2=index2+1
                else:
                    print("Image filtered")
                    print("CLIP predicted class: ",indices[0])
                    print("correct class: ",index)
            else:
                image_path = os.path.join(
                        savedir,
                        "images",
                        f"{imagenet_class_ids_local[index]:03d}",
                        f"{index2:03d}.png",
                    )
                image.save(image_path)
                index2=index2+1
    state.wait_for_everyone()
