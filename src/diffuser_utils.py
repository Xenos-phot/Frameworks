
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverSinglestepScheduler
from safetensors.torch import load_file
from diffusers import AutoencoderKL

import json
import os
from time import time
import torch

def  create_pipe():

    # Schedulers
    schedulers = {
        "Euler_K": (EulerDiscreteScheduler, {"use_karras_sigmas": True}),

        "DPMPP_2M": (DPMSolverMultistepScheduler, {}),
        "DPMPP_2M_K": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
        "DPMPP_2M_Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True}),
        "DPMPP_2M_Stable": (DPMSolverMultistepScheduler, {"euler_at_final": True}),

        "DPMPP_2M_SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_K": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_Lu": (DPMSolverMultistepScheduler, {"use_lu_lambdas": True, "algorithm_type": "sde-dpmsolver++"}),
        "DPMPP_2M_SDE_Stable": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++", 
                                                              "use_lu_lambdas": True,
                                                              "use_karras_sigmas": True,
                                                              "euler_at_final": True}),
        "Base": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
        "Base_upgrade": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True,
                                                "use_lu_lambdas": True,
                                                "euler_at_final": True})
    }
    scheduler = schedulers["Base_upgrade"][0].from_pretrained(
                "SG161222/RealVisXL_V4.0_Lightning",
                subfolder="scheduler",
                **schedulers["Base_upgrade"][1],
            )


    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16)
    model_loading_time = time()
    pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0_Lightning",
                                                     vae=vae,
                                                        torch_dtype=torch.float16,
                                                        variant = "fp16",
                                                        requires_grad=False)
    pipe.scheduler = scheduler
    # pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # pipe.enable_xformers_memory_efficient_attention() # Increases inference time
    print('Model loaded in',time()-model_loading_time,'seconds')


    embeddings_dir = "../embeddings"

    embeddings = []
    for emb_file in os.listdir(embeddings_dir):
        if not emb_file.endswith('.pt'):
            continue
        embedding_path = os.path.join(embeddings_dir, emb_file)
        embeddings.append(embedding_path)

    pipe.load_textual_inversion(embeddings)

    return pipe



def run_pipe(pipe,prompt, num_img=1):
    negative_prompt = "(octane render, render, drawing, anime, bad photo, bad photography:1.3),\
        (worst quality, low quality, blurry:1.2),\
            (bad teeth, deformed teeth, deformed lips),\
                (bad anatomy, bad proportions:1.1),\
                    (deformed iris:1.2, deformed pupils:1.2), (deformed eyes:1.2, bad eyes:1.2), \
                        (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"
    with torch.no_grad():

        images = pipe(prompt,
                        guidance_scale=1,
                        negative_prompt= negative_prompt,
                        num_inference_steps=10,
                        strength=1,
                        width=1024,
                        height=1024,
                        num_images_per_prompt=num_img,
                        generator = torch.Generator(device='cuda').manual_seed(1),
                        # negative_prompt_embeds = embeddings 
                      ).images 
    return images 