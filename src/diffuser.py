from diffusers import DiffusionPipeline
import json
import os
from time import time
import torch

output_dir = '../output/diffusers'
os.makedirs(output_dir, exist_ok=True)

#Loading model
model_loading_time = time()
pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/realvisxl-v40-lightning", torch_dtype=torch.float16, requires_grad=False)
pipe.to("cuda")
print('Model loaded in',time()-model_loading_time,'seconds')

# Loading model
with open('../prompts.json', 'r') as f:
    prompts = json.load(f)


# Running Inference
for prompt_index in range(len(prompts)):
    torch.cuda.empty_cache()
    prompt= prompts[f"{prompt_index}"]
    
    # Create a directory for each prompt
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Generate 4 images at once for the current prompt
    
    print(prompt)
    st_time = time()
    with torch.no_grad():
        images = pipe(prompt, num_images_per_prompt=4).images  # Generate 4 images at once
    print(f'For prompt {prompt_index} time taken is:', time()-st_time)
    # Save each generated image
    for j, image in enumerate(images):
        image_path = os.path.join(prompt_dir, f"img_{j}.png")
        image.save(image_path)
        print(f"Saved image: {image_path}")