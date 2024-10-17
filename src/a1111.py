import json
import time 
import requests
import base64

prompt = "<lora:FlatStyle:1>,Flat style, a fort on a island , simple and clean , using simple vector shapes, no gradient in colors with only 10 colors with no pixelation or bluriness . The flat colors should be of same shade and no bluriness near the edges . The edges should be sharp"


negative_prompt = "bad-anatomy watermarks text blurry, gradient background , text marks"
sampler_name = "DPM++ 3M SDE"
cfg_scale = 7 
steps=20
scheduler="Karras"
# inp_url="http://3.139.23.114:8191/sdapi/v1/txt2img"
# inp_url="http://3.128.124.162:8199/sdapi/v1/txt2img"
inp_url="https://static-aws-ml1.apyhi.com/vector/sdapi/v1/txt2img"
payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "sampler_name": sampler_name,
            "scheduler" : scheduler,
            "cfg_scale": cfg_scale,
            "batch_size": 1,
            "n_iter": 1,
            "width": 1024,
            "height": 1024,
            "tiling": False            
        }
response = requests.post(
    url=inp_url,
    headers = {
        'Content-Type': 'application/json',
        'Authorization': "Basic dXNlcjphcHB5aGlnaEAzMjE="
    },
    json=payload
)

output_path = "output_5.png"
if response.status_code == 200:
    result = response.json()
    # print(result)
    output_image = result["images"][0]
    with open(output_path, "wb") as out_file:
        out_file.write(base64.b64decode(output_image))
    print("Output image saved successfully.")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)