from typing import Dict, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from diffuser_utils import create_pipe, run_pipe
import ray
from ray import serve
from starlette.requests import Request
import asyncio
from datetime import datetime
import uuid
from collections import deque
import time

# Initialize Ray
ray.init()

# Define the Pydantic models
class Item(BaseModel):
    prompt: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict] = None

# Define the deployment
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
)
class DiffusionDeployment:
    def __init__(self):
        self.pipe = create_pipe()
        self.jobs = {}
        self.queue = asyncio.Queue()
        self.is_processing = False
        
    async def start_queue_processing(self):
        if not self.is_processing:
            self.is_processing = True
            await self.process_queue()

    async def process_queue(self):
        while True:
            try:
                job_id, prompt = await self.queue.get()
                self.jobs[job_id].status = "processing"
                
                # Process the image generation
                try:
                    images = run_pipe(self.pipe, prompt)
                    self.jobs[job_id].status = "completed"
                    self.jobs[job_id].completed_at = datetime.now().isoformat()
                    self.jobs[job_id].result = {"images": images}
                except Exception as e:
                    self.jobs[job_id].status = "failed"
                    self.jobs[job_id].result = {"error": str(e)}
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing queue: {e}")
                continue

    async def generate(self, prompt: str) -> Dict:
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        self.jobs[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            created_at=datetime.now().isoformat()
        )
        
        # Add to queue
        await self.queue.put((job_id, prompt))
        
        # Start queue processing if not already started
        asyncio.create_task(self.start_queue_processing())
        
        return {"job_id": job_id}

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        return self.jobs.get(job_id)

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
    result = await handle.remote({"prompt": item.prompt})
    return result

@app.get("/status/{job_id}")
async def get_status(job_id: str) -> Dict:
    handle = serve.get_deployment("DiffusionDeployment").get_handle()
    status = await handle.get_job_status.remote(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return status

# Deployment instructions
if __name__ == "__main__":
    # Create deployment
    serve.run(deployment, name="DiffusionDeployment")
    
    # To run the FastAPI app with uvicorn:
    # uvicorn app:app --host 0.0.0.0 --port 8000