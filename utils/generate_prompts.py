"""
    Functions for generating prompts for inference and image generation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import random
from allpairspy import AllPairs
from math import ceil
from utils.labels import TEXTURES_LABELS,IMAGENET_LABELS,IMAGENET_DIR_NAMES,REVERSE_IMAGENET_LABELS,imagenet_templates,PET_LABELS,FOOD_LABELS,CAR_LABELS,FLOWER_LABELS,AIRCRAFT_LABELS,EUROSAT_LABELS,TEXTURES_LABELS,REVERSE_IMAGENET_DIR_NAMES
"""
    Utilities for ImageNet
"""
def label_to_text(label):
    return IMAGENET_LABELS[label]

def text_to_label(text):
    return REVERSE_IMAGENET_LABELS[text]

def dir_name_to_label(dir_name):
    return IMAGENET_DIR_NAMES[dir_name]

def label_to_dir_name(label):
    return REVERSE_IMAGENET_DIR_NAMES[label]

with open("./path/to/imagenet100_classes.txt", "r") as f:
    IMAGENET_100_DIR_NAMES = [line.strip() for line in f]
IMAGENET_100_LABELS = [dir_name_to_label(dirname) for dirname in IMAGENET_100_DIR_NAMES]

def get_synset(id):
    dirname = label_to_dir_name(id)
    pos = dirname[0]
    offset = int(dirname[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

def definition_prompt_imagenet(id):
    synset = get_synset(id)
    word = synset.lemmas()[0].name().replace("_", " ")
    definition = synset.definition()
    return f"{word}, {definition}"

def simple_prompt_imagenet(id):
    class_name = IMAGENET_LABELS[id]
    return class_name

def photo_prompt_imagenet(id):
    word = simple_prompt_imagenet(id)
    return f"a photo of a {word}"

def prompt_templates_imagenet(id):
    classname = simple_prompt_imagenet(id)
    # use imagenet templates from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    prompts = [template.format(classname) for template in imagenet_templates]
    return prompts

def single_prompt_templates_imagenet(id,k):
    classname = simple_prompt_imagenet(id)
    # use imagenet templates from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    specific_template = imagenet_templates[k]
    prompts = specific_template.format(classname)
    return prompts

def photo_prompt_imagenet_from_label(label):
    id = IMAGENET_DIR_NAMES[label]
    word = simple_prompt_imagenet(id)
    return f"a photo of a {word}"
"""
    Utilities for Oxford Pets
"""
def simple_prompt_pets(id):
    class_name = PET_LABELS[id]
    return class_name

def photo_prompt_pets(id):
    class_name = PET_LABELS[id]
    prompt = "a photo of a " + class_name
    return prompt
    
"""
    Utilities for Food-101
"""
def simple_prompt_food(id):
    class_name = FOOD_LABELS[id+1]
    return class_name

def photo_prompt_food(id):
    class_name = FOOD_LABELS[id+1]
    prompt = "a photo of a " + class_name
    return prompt

"""
    Utilities for Stanford-Cars
"""
def simple_prompt_cars(id):
    class_name = CAR_LABELS[id]
    return class_name

def photo_prompt_cars(id):
    class_name = CAR_LABELS[id]
    prompt = "a photo of a " + class_name
    return prompt

"""
    Utilities for Flowers-102
"""
def simple_prompt_flowers(id):
    class_name = FLOWER_LABELS[id]
    return class_name

def photo_prompt_flowers(id):
    class_name = FLOWER_LABELS[id]
    prompt = "a photo of a " + class_name
    return prompt
"""
    Utilities for FGVC-Aircraft
"""
def simple_prompt_aircraft(id):
    class_name = AIRCRAFT_LABELS[id]
    return class_name

def photo_prompt_aircraft(id):
    class_name = AIRCRAFT_LABELS[id]
    prompt = "a photo of a " + class_name
    return prompt
"""
    Utilities for EUROSAT
"""
def simple_prompt_eurosat(id):
    class_name = EUROSAT_LABELS[id]
    return class_name

def photo_prompt_eurosat(id):
    class_name = AIRCRAFT_LABELS[id]
    prompt = "a centered satellite photo of a " + class_name
    return prompt
"""
    Utilities for Describable Textures
"""
def simple_prompt_texture(id):
    class_name = TEXTURES_LABELS[id]
    return class_name

def photo_prompt_texture(id):
    class_name = TEXTURES_LABELS[id]
    prompt = "a photo of a " + class_name + " texture"
    return prompt
"""
    Simple prompt generators.
"""
def simple_prompt(dataset,id):
    if dataset=="pets":
        prompt=simple_prompt_pets(id)
    elif dataset=="food":
        prompt=simple_prompt_food(id)
    elif dataset=="flowers":
        prompt=simple_prompt_flowers(id)
    elif dataset=="cars":
        prompt=simple_prompt_cars(id)
    elif dataset=="aircraft":
        prompt=simple_prompt_aircraft(id)
    elif dataset=="texture":
        prompt=simple_prompt_texture(id)
    elif dataset=="imagenet":
        prompt=simple_prompt_imagenet(id)
    else:
        print("No suitable dataset selected")
    return prompt

def photo_prompt(dataset,id):
    if dataset=="pets":
        prompt=photo_prompt_pets(id)
    elif dataset=="food":
        prompt=photo_prompt_food(id)
    elif dataset=="imagenet":
        prompt=photo_prompt_imagenet(id)
    elif dataset=="flowers":
        prompt=photo_prompt_flowers(id)
    elif dataset=="cars":
        prompt=photo_prompt_cars(id)
    elif dataset=="aircraft":
        prompt=photo_prompt_aircraft(id)
    elif dataset=="texture":
        prompt=photo_prompt_texture(id)
    elif dataset=="eurosat":
        prompt=photo_prompt_eurosat(id)
    else:
        print("No suitable dataset selected")
    return prompt

def inference_prompts(dataset,id):
    if dataset=="pets":
        class_name = PET_LABELS[id]
        prompt = "a photo of a " + class_name + ", a type of pet."
    elif dataset=="food":
        class_name = FOOD_LABELS[id+1]
        prompt = "a photo of a " + class_name + ", a type of food."
    elif dataset=="flowers":
        class_name = FLOWER_LABELS[id]
        prompt = "a photo of a " + class_name + ", a type of flower."
    elif dataset=="texture":
        class_name = TEXTURES_LABELS[id]
        prompt = "a photo of a " + class_name + " texture."
    elif dataset=="cars":
        class_name = CAR_LABELS[id]
        prompt = "a photo of a " + class_name + ", a type of car."
    elif dataset=="aircraft":
        class_name = AIRCRAFT_LABELS[id]
        prompt = "a photo of a " + class_name + ", a type of aircraft."
    elif dataset=="eurosat":
        class_name = EUROSAT_LABELS[id]
        prompt = "a centered satellite photo of a " + class_name
    elif dataset=="imagenet":
        prompt = "a photo of a " + simple_prompt_imagenet(id)
    else:
        print("No suitable dataset selected")
    return prompt

"""
    Create diverse captions based on location, time of day, position and camera angle using one call to the LLM for every option.
"""
def create_locations(N,class_name,LLM,tokenizer):
    """Sample options for location."""
    messages = [
        {"role": "user", "content": 
        "Where could a photo of a "+class_name+" be taken? Output only "+str(N+1)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = LLM.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    splitted_test = response_text.split("<|assistant|>")
    caption_with_indent=splitted_test[1]
    # remove indent and "" signs at the beginning and end if present
    response=caption_with_indent[1:]
    response=response.replace('"', '')
    response_splitted=response.split(";")[0]
    locations=response_splitted.splitlines()
    locations = [o[3:] for o in locations]
    return locations[0:N]

def create_daytimes(N,class_name,LLM,tokenizer):
    """Sample options for daytime."""
    messages = [
        {"role": "user", "content": 
        "At what daytime could a photo of a "+class_name+" be taken? Output only "+str(N+1)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = LLM.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    splitted_test = response_text.split("<|assistant|>")
    caption_with_indent=splitted_test[1]
    # remove indent and "" signs at the beginning and end if present
    response=caption_with_indent[1:]
    response=response.replace('"', '')
    response_splitted=response.split(";")[0]
    daytimes=response_splitted.splitlines()
    daytimes = [o[3:] for o in daytimes]
    return daytimes[0:N]

def create_positions(N,class_name,LLM,tokenizer):
    """Sample options for position of the main object."""
    messages = [
        {"role": "user", "content": 
        "In which position could "+class_name+" be in a photo of a "+class_name+"? Output only "+str(N+1)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = LLM.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    splitted_test = response_text.split("<|assistant|>")
    caption_with_indent=splitted_test[1]
    # remove indent and "" signs at the beginning and end if present
    response=caption_with_indent[1:]
    response=response.replace('"', '')
    response_splitted=response.split(";")[0]
    positions=response_splitted.splitlines()
    positions = [o[3:] for o in positions]
    return positions[0:N]

def create_angles(N,class_name,LLM,tokenizer):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "From which camera angles could a photo of a "+class_name+" be taken? Output only "+str(N+1)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    # for i in tqdm(range(8)):
    outputs = LLM.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    splitted_test = response_text.split("<|assistant|>")
    caption_with_indent=splitted_test[1]
    # remove indent and "" signs at the beginning and end if present
    response=caption_with_indent[1:]
    response=response.replace('"', '')
    response_splitted=response.split(";")[0]
    angles=response_splitted.splitlines()
    angles = [o[3:] for o in angles]
    return angles
"""
    Calls to LLama
"""
def llama_superclass(dataset,class_ids,LLM_model_id,LLM_local,tokenizer_local):
    # overall attribute collections
    llama_superclasses=[]
    if dataset=="pets":
        superclass_name="pet"
    elif dataset=="food":
        superclass_name="food"
    elif dataset=="flowers":
        superclass_name="flower"
    elif dataset=="texture":
        superclass_name="texture"
    elif dataset=="cars":
        superclass_name="car"
    elif dataset=="aircraft":
        superclass_name="aircraft"
    else:
        superclass_name="thing"
    for index in tqdm(range(len(class_ids))):
        messages = [
        {"role": "user", "content": 
            "What kind of "+superclass_name+" is a "+simple_prompt(dataset,class_ids[index])+".  Output only one word which is a noun and not a description."},
        ]
        prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = LLM_local.generate(prompt, max_new_tokens=32, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
        splitted_test = response_text.split("[/INST]")
        superclass=splitted_test[1]
        superclass=superclass[2:]
        llama_superclasses = llama_superclasses+[superclass]
    return llama_superclasses

def create_locations_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for location."""
    messages = [
        {"role": "user", "content": 
        "Where could a photo of a "+class_name+" ("+superclass_name+") be taken? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    locations_global=[]
    for i in range(n_repeats):
        locations=[]
        while (len(locations)!=5):
            print("Rejected locations due to not enough options, repeating...")
            print(locations)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            locations=caption_with_indent.splitlines()
            locations = [o[3:] for o in locations]
            locations = locations[2:7]
        locations_global = locations_global + locations
    print(locations_global)
    return locations_global

def create_daytimes_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for daytime."""
    messages = [
        {"role": "user", "content": 
        "At what daytime could a photo of a "+class_name+" ("+superclass_name+") be taken? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    daytimes_global=[]
    for i in range(n_repeats):
        daytimes=[]
        while (len(daytimes)!=5):
            print("Rejected daytimes")
            print(daytimes)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            daytimes=caption_with_indent.splitlines()
            daytimes = [o[3:] for o in daytimes]
            daytimes = daytimes[2:7]
        daytimes_global = daytimes_global + daytimes
    print(daytimes_global)
    return daytimes_global

def create_positions_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for position of the main object."""
    messages = [
        {"role": "user", "content": 
        "In which position could "+class_name+" ("+superclass_name+") be in a photo of a "+class_name+" ("+superclass_name+") ? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    positions_global=[]
    for i in range(n_repeats):
        positions=[]
        while (len(positions)!=5):
            print("Rejected positions")
            print(positions)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            positions=caption_with_indent.splitlines()
            positions = [o[3:] for o in positions]
            positions = positions[2:7]
        positions_global = positions_global + positions
    print(positions_global)
    return positions_global

def create_angles_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "From which camera angles could a photo of a "+class_name+" ("+superclass_name+") be taken? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    angles_global=[]
    for i in range(n_repeats):
        angles=[]
        while (len(angles)!=5):
            print("Rejected angles")
            print(angles)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            angles=caption_with_indent.splitlines()
            angles = [o[3:] for o in angles]
            angles = angles[2:7]
        angles_global = angles_global + angles
    print(angles_global)
    return angles_global

def create_color_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "What color could a "+class_name+" ("+superclass_name+") have? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    colors_global=[]
    for i in range(n_repeats):
        colors=[]
        while (len(colors)!=5):
            print("Rejected colors")
            print(colors)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            colors=caption_with_indent.splitlines()
            colors = [o[3:] for o in colors]
            colors = colors[2:7]
        colors_global = colors_global + colors
    print(colors_global)
    return colors_global

def create_servings_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local,superclass_llama):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "How could a "+class_name+" ("+superclass_llama+", "+superclass_name+") be served? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    colors_global=[]
    for i in range(n_repeats):
        colors=[]
        while (len(colors)!=5):
            print("Rejected colors")
            print(colors)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            colors=caption_with_indent.splitlines()
            colors = [o[3:] for o in colors]
            colors = colors[2:7]
        colors_global = colors_global + colors
    print(colors_global)
    return colors_global

def create_fabric_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "Which fabric can a "+class_name+" texture be made of? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    colors_global=[]
    for i in range(n_repeats):
        colors=[]
        while (len(colors)!=5):
            print("Rejected colors")
            print(colors)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            colors=caption_with_indent.splitlines()
            colors = [o[3:] for o in colors]
            colors = colors[2:7]
        colors_global = colors_global + colors
    print(colors_global)
    return colors_global

def create_object_llama_repeated(n_repeats,class_name,superclass_name,LLM_local,tokenizer_local):
    """Sample options for the camera angle."""
    messages = [
        {"role": "user", "content": 
        "Which object can a "+class_name+" texture be part of? Output only "+str(6)+" numbered bullet points without complete sentences and no explanations."},
    ]
    prompt = tokenizer_local.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    colors_global=[]
    for i in range(n_repeats):
        colors=[]
        while (len(colors)!=5):
            print("Rejected colors")
            print(colors)
            outputs = LLM_local.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            response_text = tokenizer_local.batch_decode(outputs, skip_special_tokens=True)[0]
            splitted_test = response_text.split("[/INST]")
            caption_with_indent=splitted_test[1]
            colors=caption_with_indent.splitlines()
            colors = [o[3:] for o in colors]
            colors = colors[2:7]
        colors_global = colors_global + colors
    print(colors_global)
    return colors_global

def llama_attributes_repeated(dataset,class_ids,LLM_model_id,max_n_tokens,temp,k,p,savedir,n_repeats):
    checkpoint = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(checkpoin)
    LLM = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto") 
    # get precise superclasses from LLM
    superclasses_llama=llama_superclass(dataset,class_ids,LLM_model_id,LLM,tokenizer)
    # overall attribute collections
    llama_locations=[]
    llama_daytimes=[]
    llama_positions=[]
    llama_angles=[]
    if dataset=="pets":
        superclass_name="pet"
    elif dataset=="food":
        superclass_name="food"
    elif dataset=="flowers":
        superclass_name="flower"
    elif dataset=="texture":
        superclass_name="texture"
    elif dataset=="cars":
        superclass_name="car"
    elif dataset=="aircraft":
        superclass_name="aircraft"
    else:
        superclass_name=""
    for index in tqdm(range(len(class_ids))):
        class_name = simple_prompt(dataset,class_ids[index])
        # Collect first three contextual dimensions
        if dataset=="texture":
            locations=create_locations_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
            daytimes=create_fabric_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
            angles=create_object_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        else:
            locations=create_locations_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
            daytimes=create_daytimes_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
            angles=create_angles_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        # Collect fourth contextual dimension
        if dataset=="pets":
            positions=create_positions_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        elif dataset=="food":
            positions=create_servings_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer,superclasses_llama[index])
        elif dataset=="flowers":
            positions=create_color_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        elif dataset=="cars":
            positions=create_color_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        elif dataset=="texture":
            positions=create_color_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        elif dataset=="aircraft":
            positions=create_color_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        else:
            positions=create_positions_llama_repeated(n_repeats,class_name,superclass_name,LLM,tokenizer)
        
        # append to overall storage
        llama_locations.append(locations)
        llama_daytimes.append(daytimes)
        llama_positions.append(positions)
        llama_angles.append(angles)
    return superclasses_llama,llama_locations,llama_daytimes,llama_positions,llama_angles
"""
    Create diverse captions based on location, time of day, position and camera angle using one call to the LLM for every option.
"""
def get_all_pairs_length(options_per_attribute):
    parameter_ids = [range(options_per_attribute),range(options_per_attribute),range(options_per_attribute),range(options_per_attribute)]
    N=0
    for i, pairs in enumerate(AllPairs(parameter_ids)):
        N+=1
    return N

def generate_prompts_all_pairs_repeated(dataset,class_ids,LLM_model_id,max_n_tokens,temp,k,p,savedir,options_per_attribute):
    """Create cations based on the above sampled attributes."""
    for class_id in class_ids:
        path = os.path.join(str(savedir[0]), "captions", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
        path = os.path.join(str(savedir[0]), "properties", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
    properties_path = os.path.join(savedir[0],"properties")
    os.makedirs(properties_path, exist_ok=True)
    # check how many times to repeat asking for 5 options per attribute
    n_repeats = ceil(options_per_attribute/5)
    pair_ids = AllPairs([range(5*n_repeats),range(5*n_repeats),range(5*n_repeats),range(5*n_repeats)])
    pair_ids_list=[]
    for i, pairs in enumerate(pair_ids):
        pair_ids_list.append(pairs)
    prompts_per_class=len(pair_ids_list)
    diverse_prompts=["" for x in range(len(class_ids)*prompts_per_class)]
    print("Prompts per class: ",prompts_per_class)
    # Note: for cars, flowers, food positions actually corresponds to colors
    llama_superclasses, llama_locations,llama_daytimes,llama_positions,llama_angles=llama_attributes_repeated(dataset,class_ids,LLM_model_id,max_n_tokens,temp,k,p,savedir,n_repeats)
    
    for index in tqdm(range(len(class_ids))):
        class_name = simple_prompt(dataset,class_ids[index])
        locations = llama_locations[index]
        locations_file = open(os.path.join(properties_path,f"{class_ids[index]:03d}","locations.txt"), "w")
        for location in locations:
            locations_file.write(location)
            locations_file.write("\n")
        locations_file.close()
        daytimes = llama_daytimes[index]
        daytimes_file = open(os.path.join(properties_path,f"{class_ids[index]:03d}","daytimes.txt"), "w")
        for daytime in daytimes:
            daytimes_file.write(daytime)
            daytimes_file.write("\n")
        daytimes_file.close()
        positions = llama_positions[index]
        positions_file = open(os.path.join(properties_path,f"{class_ids[index]:03d}","positions.txt"), "w")
        for position in positions:
            positions_file.write(position)
            positions_file.write("\n")
        positions_file.close()
        angles = llama_angles[index]
        angles_file = open(os.path.join(properties_path,f"{class_ids[index]:03d}","angles.txt"), "w")
        for angle in angles:
            angles_file.write(angle)
            angles_file.write("\n")
        angles_file.close()
        # Create all pairs captions, only use first options_per_attribute entries of the attributes to neglect any additional responses from the LLM
        attribute_collection = [locations, daytimes, positions, angles]
        parameter_pairs = AllPairs(attribute_collection)
        for index2, pairs in enumerate(parameter_pairs):
            print("Index in all pair list:", index2)
            print("Content", pairs)
            # random draw from the options, ++ is used for prompt weighting (https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts)
            if dataset=="pets":
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a " + class_name+"++++ ("+llama_superclasses[index]+"++, pet), " + pairs[0] + ", " + pairs[1] + ", " + pairs[2] + ", " + pairs[3]
            elif dataset=="food":
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a (" + class_name + ")1.2, (" + pairs[2] + ")1.5, Side angle," + " background is a " + pairs[0] + ", background is not blurred, (" + pairs[3] + ")0.1, (" + pairs[1] + ")0.1"
            elif dataset=="flowers":
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a "+ class_name + "++ (a" +llama_superclasses[index]+ ", flower), close-up shot of the blossom, " + pairs[0] + ", " + pairs[3] + ", (" + pairs[2] + ")0.1, (" + pairs[1] + ")0.1"
            elif dataset=="cars":
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a " + pairs[2] + " " + class_name+"++++ ("+llama_superclasses[index]+"--, car), " + pairs[0] + ", " + pairs[1] + ", " + pairs[3]
            elif dataset=="aircraft":
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a " + pairs[2] + " " + class_name+"+++ ("+llama_superclasses[index]+"----, aircraft), " + pairs[0] + "+, " + pairs[3] + "+, " + pairs[1]
            elif dataset=="texture":
                diverse_prompts[index*prompts_per_class+index2] =  "A photo of a " + pairs[2] + "- " + class_name+"++++ texture+++, close-up shot, " + pairs[1] + "--, " + pairs[3] + "---, " + pairs[0] + "---- " 
            elif dataset=="imagenet": 
                k_temp = random.randint(0,79)
                specific_template = imagenet_templates[k_temp]
                diverse_prompts[index*prompts_per_class+index2] = str(specific_template.format(class_name))+"++++ , " + pairs[0] + ", " + pairs[1] + ", " + pairs[2] + ", " + pairs[3]
                print(diverse_prompts[index*prompts_per_class+index2])
            else:
                diverse_prompts[index*prompts_per_class+index2] = "A photo of a " + class_name+"++++ , " + pairs[0] + ", " + pairs[1] + ", " + pairs[2] + ", " + pairs[3]
            caption_path = os.path.join(
                    savedir[0],
                    "captions",
                    f"{class_ids[index]:03d}",
                    f"{index2:03d}.txt",
                )
            text_file = open(caption_path, "w")
            text_file.write(diverse_prompts[index*prompts_per_class+index2])
            text_file.close()
    torch.cuda.empty_cache()
    return diverse_prompts, prompts_per_class

def generate_simple_prompts(dataset,class_ids,prompts_per_class,savedir):
    """Create simple captions with -a photo of-"""
    for class_id in class_ids:
        path = os.path.join(savedir[0], "captions", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
    simple_prompts=["" for x in range(len(class_ids)*prompts_per_class)]
    
    for index in range(len(class_ids)):
        for index2 in tqdm(range(prompts_per_class)):
            simple_prompts[index*prompts_per_class+index2]=photo_prompt(dataset,class_ids[index])
            
            caption_path = os.path.join(
                    savedir[0],
                    "captions",
                    f"{class_ids[index]:03d}",
                    f"{index2:03d}.txt",
                )
            text_file = open(caption_path, "w")
            text_file.write(simple_prompts[index*prompts_per_class+index2])
            text_file.close()
    torch.cuda.empty_cache()
    return simple_prompts


"""
    Obtain all or specific prompts for certain classes.
"""
def get_IN_class_prompts(class_ids):
    """Get all ImageNet classes."""
    IN_prompts=[photo_prompt_imagenet(class_ids[index]) for index in range(1000)]
    return IN_prompts

def get_dedicated_IN_class_names(class_ids):
    """Get names of specific ImageNet classes."""
    IN_prompts=[simple_prompt_imagenet(index) for index in class_ids]
    return IN_prompts

def get_dedicated_class_names(dataset,class_ids):
    """Get names of specific classes."""
    if dataset=="imagenet":
        IN_prompts=[simple_prompt_imagenet(index) for index in class_ids]
    elif dataset=="pets":
        IN_prompts=[simple_prompt_pets(index) for index in class_ids]
    elif dataset=="flowers":
        IN_prompts=[simple_prompt_flowers(index) for index in class_ids]
    elif dataset=="texture":
        IN_prompts=[simple_prompt_flowers(index) for index in class_ids]
    elif dataset=="cars":
        IN_prompts=[simple_prompt_cars(index) for index in class_ids]
    elif dataset=="aircraft":
        IN_prompts=[simple_prompt_cars(index) for index in class_ids]
    elif dataset=="food":
        IN_prompts=[simple_prompt_food(index) for index in class_ids]
    return IN_prompts

"""
    Read existing captions
"""
def read_captions(class_ids,prompts_per_class,savedir):
    """Create cations based on the above sampled attributes."""
    for class_id in class_ids:
        path = os.path.join(savedir[0], "captions", f"{class_id:03d}")
        os.makedirs(path, exist_ok=True)
    prompts=["" for x in range(len(class_ids)*prompts_per_class)]
    for index in tqdm(range(len(class_ids))):     
        for index2 in range(prompts_per_class): 
            caption_path = os.path.join(
                    savedir[0],
                    "captions",
                    f"{class_ids[index]:03d}",
                    f"{index2:03d}.txt",
                )
            text_file = open(caption_path, "r")
            prompts[index*prompts_per_class+index2] = text_file.read()
            text_file.close()
    return prompts
"""
    Remove weighting from prompts
"""
def remove_weighting(s):
    target_string1 = s.replace('-', '')
    target_string2 = target_string1.replace('-', '')
    return target_string2
