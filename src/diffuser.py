from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, LCMScheduler, DPMSolverSinglestepScheduler
from safetensors.torch import load_file
from diffusers import AutoencoderKL

import json
import os
from time import time
import torch

from diffuser_utils import create_pipe, run_pipe

output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# Loading data
with open('../prompts.json', 'r') as f:
    prompts = json.load(f)

pipe =create_pipe()

average_time =0
# Running Inference
for prompt_index in range(len(prompts)):
    torch.cuda.empty_cache()
    prompt= prompts[prompt_index]
    
    # Create a directory for each prompt
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)
    
    print(prompt)
    st_time = time()

    images = run_pipe(pipe,prompt)
        
    print(f'For prompt {prompt_index} time taken is:', time()-st_time)
    average_time+=time()-st_time

    # Save each generated image
    for j, image in enumerate(images):
        image_path = os.path.join(prompt_dir, f"diffuser_{j}.png")
        image.save(image_path)
        print(f"Saved image :{image_path}")

print(f"Average_time:", average_time/len(prompts))