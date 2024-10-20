import json
import time 
import requests
import base64
import os

# prompt = "generate a high resolution image for the car booking website banner generate animated pictures with a map, high quality, detailed"
with open('../prompts.json', 'r') as f:
    prompts = json.load(f)

negative_prompt= "(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"
sampler_name = "DPM++ SDE"
cfg_scale = 7
steps=6
scheduler="Karras"

inp_url="http://127.0.0.1:7860/sdapi/v1/txt2img"
payload = {
            "prompt": "",
            "negative_prompt": negative_prompt,
            "steps": steps,
            "sampler_name": sampler_name,
            "scheduler" : scheduler,
            "cfg_scale": cfg_scale,
            "batch_size": 4,
            "n_iter": 1,
            "width": 1024,
            "height": 1024,
            "tiling": False            
        }

output_dir = "../output"
average_time =0
for prompt_index in range(len(prompts)):
    prompt= prompts[f"{prompt_index}"]
    payload["prompt"] = prompt
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)
    
    request_sent_time = time.time()
    response = requests.post(
        url=inp_url,
        json=payload
    )
    print(f"Response for prompt {prompt_index} received in {time.time()-request_sent_time} seconds")
    
    average_time += time.time()-request_sent_time
    
    if response.status_code == 200:
        result = response.json()
        output_images = result["images"]
        for i, output_image in enumerate(output_images):
            output_path =os.path.join(prompt_dir,f"a1111_{i}.png")
            with open(output_path, "wb") as out_file:
                out_file.write(base64.b64decode(output_image))
            print(f"Output image saved successfully for prompt {prompt_index}_{i}.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        
print(f"Average time taken for each prompt for batch size of {payload['batch_size']} is {average_time/len(prompts)} seconds")