from typing import Union, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from diffuser import create_pipe, run_pipe
import ray
from ray import serve
from ray.serve.handle import RayServeHandle
from starlette.requests import Request

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
        return {"status": "success", "images": images}
    
    async def __call__(self, request: Request) -> Dict:
        json_body = await request.json()
        return await self.generate(json_body["prompt"])

# Create FastAPI app
app = FastAPI()

# Initialize Ray Serve
serve.start(detached=True)

# Deploy the model
deployment = DiffusionDeployment.bind()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate(item: Item) -> Dict:
    handle = serve.get_deployment("DiffusionDeployment").get_handle()
    return await handle.remote({"prompt": item.prompt})

# Deployment instructions
if __name__ == "__main__":
    # Create deployment
    serve.run(deployment, name="DiffusionDeployment")
    
    # To run the FastAPI app with uvicorn:
    # uvicorn app:app --host 0.0.0.0 --port 8000