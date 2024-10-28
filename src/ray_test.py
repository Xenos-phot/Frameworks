from typing import Union, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from diffuser_utils import create_pipe, run_pipe
import ray
from ray import serve
from starlette.requests import Request
import json
# Initialize Ray
ray.init()

# Define the Pydantic model
class Item(BaseModel):
    prompt: str

# Define the deployment
@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class DiffusionDeployment:
    def __init__(self):
        self.pipe = create_pipe()
    
    async def generate(self, prompt: str) -> Dict:
        images = run_pipe(self.pipe, prompt)
        return {"status": "success"}
    
    async def __call__(self, request: Request) -> Dict:
        path = request.url.path
        method = request.method
        
        if path == "/" and method =='GET':
            return {"status": "success"}
        elif path == "/generate" and method == "POST":
            json_body = await request.json()
            
            if "prompt" not in json_body:
                
                return json.dumps({"error": "prompt is required"})
            return await self.generate(json_body["prompt"])


# Initialize Ray Serve
serve.start(detached=True)

# Deploy the model
deployment = DiffusionDeployment.bind()

