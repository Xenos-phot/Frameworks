from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler
from safetensors.torch import load_file
from diffusers import AutoencoderKL

import json
import os
from time import time
import torch

def  create_pipe():

    # Schedulers
    common_config = {'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear'}
    schedulers = {
        "Euler_K": (EulerDiscreteScheduler, {"use_karras_sigmas": True}),

        "DPMPP_2M": (DPMSolverMultistepScheduler, {}),
        "DPMPP_2M_K": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
        "DPMPP_2M_Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True}),
        "DPMPP_2M_Stable": (DPMSolverMultistepScheduler, {"euler_at_final": True}),

        "DPMPP_2M_SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_K": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_Stable": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", "euler_at_final": True}),
    }
    scheduler = schedulers["DPMPP_2M_SDE_Lu"][0].from_pretrained(
                "SG161222/RealVisXL_V4.0_Lightning",
                subfolder="scheduler",
                **schedulers["DPMPP_2M_SDE_Lu"][1],
            )

    # seed 
    seed =1
    generator = torch.Generator(device='cuda').manual_seed(seed)


    model_loading_time = time()
    pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0_Lightning",
                                                        torch_dtype=torch.float16,
                                                        variant = "fp16",
                                                        requires_grad=False)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    # pipe.enable_xformers_memory_efficient_attention() # Increases inference time
    print('Model loaded in',time()-model_loading_time,'seconds')

    return pipe


def run_pipe(pipe,prompt):
    with torch.no_grad():

        images = pipe(prompt,
                        guidance_scale=1,
                        negative_prompt="(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured",
                        num_inference_steps=10,
                        strength=1,
                        width=1024,
                        height=1024,
                        num_images_per_prompt=4,
                      ).images 
    return images 

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
    prompt= prompts[f"{prompt_index}"]
    
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