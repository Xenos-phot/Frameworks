from typing import Union, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from diffuser_utils import create_pipe, run_pipe
import ray
from ray import serve
from starlette.requests import Request
import json

from PIL import Image
import io
import base64
def convert_to_b64(img):
    # Create a BytesIO object to hold the image data
    buffered = io.BytesIO()
    # Save the image to the BytesIO object in a specific format (e.g., PNG)
    img.save(buffered, format="PNG")
    # Get the byte data
    img_byte = buffered.getvalue()
    # Encode the byte data to base64
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    return img_base64


# Initialize Ray
# Only initialize Ray if it’s not already running
print(ray.is_initialized())
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

# Define the Pydantic model
class Item(BaseModel):
    prompt: str

# Define the deployment
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class DiffusionDeployment:
    def __init__(self):
        self.pipe = create_pipe()
    
    async def generate(self, prompt: str) -> Dict:
        images = run_pipe(self.pipe, prompt,num_img=1)
        images_url =[]
        for i,img in enumerate(images):
            img.save(f'/root/Frameworks/output/{i}',"webp")
            images_url.append(f"/root/Frameworks/output/{i}.png")
        # b64_images = [convert_to_b64(img) for img in images]
        # return {"status": "Success got the result", "images": b64_images, "type": "b64" }
        return {"status": "Success got the result", "images": images_url, "type": "url" }
    
    async def __call__(self, request: Request) -> Dict:
        path = request.url.path
        method = request.method
        print(path, method)
        if path == "/" and method =='GET':
            return {"status": "Home Page"}
        elif path == "/generate" and method == "POST":
            json_body = await request.json()
            
            if "prompt" not in json_body:
                
                return json.dumps({"error": "prompt is required"})
            return await self.generate(json_body["prompt"])


# Initialize Ray Serve
serve.start(detached=True)

# Deploy the model
deployment = DiffusionDeployment.bind()

