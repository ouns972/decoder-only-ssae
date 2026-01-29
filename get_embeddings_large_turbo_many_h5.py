# get multiple embeddings from the text encoder (for SAE)

import os
import h5py
import json
import time
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import numpy as np
import random
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, AutoTokenizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    prompts_filename = "prompts/your_directory/prompts.json"

    folder = "prompts/your_directory/embds/"
    
    with open(prompts_filename, "r") as f:
        prompts = json.load(f)

    print("len = ", len(prompts))

    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    #style = ". Comic book style. graphic illustration, comic art, graphic novel art, vibrant, highly detailed"
    style = ""

    t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16,
        device_map = "balanced"
    )

    #pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    n = 4000

    for (i, prompt) in enumerate(prompts[:n]):
        # without style for the moment

        prompt = prompt["prompt"]

        prompt = prompt + style
        
        if i % 1 == 0:
            print("i : ", i, prompt)

        setup_seed(0) # not necessary
        with torch.no_grad():
            embds = pipeline.encode_prompt(prompt, prompt, prompt)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embds

        folder_embds_i = folder + f"embds_{i}/"
        os.makedirs(folder_embds_i, exist_ok=True)
        
        with open(folder_embds_i + "prompts.txt", "w") as file:
            file.write(prompt)

        data = prompt_embeds.to(torch.float16).flatten().cpu().detach().numpy()
        with h5py.File(folder_embds_i + "embds.h5", "w") as f:
            f.create_dataset("vector", data=data)

        data2 = pooled_prompt_embeds.to(torch.float16).flatten().cpu().detach().numpy()
        with h5py.File(folder_embds_i + "embds_pooled.h5", "w") as f:
            f.create_dataset("vector", data=data2)
    
    