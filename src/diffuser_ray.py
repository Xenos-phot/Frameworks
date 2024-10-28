import ray
from ray import serve
import asyncio
from typing import Dict, Optional
from datetime import datetime
import uuid
import json
from http import HTTPStatus
import time
from diffuser_utils import create_pipe, run_pipe
import signal
import sys
from asyncio import Lock
# Initialize Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)
class JobStatus:
    def __init__(self, job_id: str, status: str):
        self.job_id = job_id
        self.status = status
        self.created_at = datetime.now().isoformat()
        self.completed_at = None
        self.result = None
    
    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result
        }

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    route_prefix="/"
)
class DiffusionDeployment:
    def __init__(self):
        self.pipe = create_pipe()
        self.jobs = {}
        self.queue = asyncio.Queue(maxsize=100)  # Queue with max size 100
        self.is_processing = False
        self.max_concurrent_jobs = 4
        self.current_jobs = 0
        self.semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
    async def start_queue_processing(self):
        if not self.is_processing:
            self.is_processing = True
            asyncio.create_task(self.process_queue())

    async def process_queue(self):
        while True:
            try:
                async with self.semaphore:
                    job_id, prompt = await self.queue.get()
                    self.current_jobs += 1
                    self.jobs[job_id].status = "processing"
                    
                    try:
                        images = run_pipe(self.pipe, prompt)
                        self.jobs[job_id].status = "completed"
                        self.jobs[job_id].completed_at = datetime.now().isoformat()
                        self.jobs[job_id].result = {"images": images}
                    except Exception as e:
                        self.jobs[job_id].status = "failed"
                        self.jobs[job_id].result = {"error": str(e)}
                    finally:
                        self.current_jobs -= 1
                        self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing queue: {e}")
                await asyncio.sleep(1)
                continue

    async def generate(self, prompt: str) -> Dict:
        if self.queue.full():
            return {"error": "Queue is full, please try again later"}
            
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = JobStatus(job_id=job_id, status="queued")
        
        try:
            await asyncio.wait_for(self.queue.put((job_id, prompt)), timeout=5.0)
            await self.start_queue_processing()
            return {"job_id": self.jobs[job_id]}
        except asyncio.TimeoutError:
            del self.jobs[job_id]
            return {"error": "Queue operation timed out"}

    async def handle_request(self, request) -> Dict:
        request_path = request.url.path
        method = request.method
        path = request_path

        try:
            if path == "/" and method == "GET":
                return await self.handle_root()
            
            elif path == "/generate" and method == "POST":
                body = await request.json()
                if "prompt" not in body:
                    return json.dumps({"error": "prompt is required"})
                result = await self.generate(body["prompt"])
                
                if isinstance(result, tuple) and len(result) == 2:
                    return json.dumps(result[0])
                    
                return json.dumps(result)
                
            
            elif path.startswith("/status/") and method == "GET":
                job_id = path.split("/")[-1]
                return await self.handle_status(job_id)
            
            elif path == "/queue-info" and method == "GET":
                return await self.handle_queue_info()
            
            else:
                return json.dumps({"error": "Not found"})
                

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def handle_root(self):
        return json.dumps({
                "status": "healthy",
                "version": "1.0.0"
            })

    async def handle_status(self, job_id: str):
        if job_id not in self.jobs:
            return json.dumps({"error": "Job not found"})
        
        return json.dumps(self.jobs[job_id].to_dict())

    async def handle_queue_info(self):
        info = {
            "queue_size": self.queue.qsize(),
            "queue_remaining": self.queue.maxsize - self.queue.qsize(),
            "is_processing": self.is_processing,
            "active_jobs": self.current_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "total_jobs": len(self.jobs)
        }
        return json.dumps(info)

    async def __call__(self, request) -> Dict:
        return await self.handle_request(request)

def signal_handler(signal, frame):
    print("Shutting down gracefully...")
    serve.shutdown()
    ray.shutdown()
    sys.exit(0)


try:
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Start Ray Serve
    serve.start(detached=True)
    
    # Deploy the service
    deployment = DiffusionDeployment.bind()
    print("Running...")
    # serve.run(deployment)

    print("Service is running! You can access it at: http://localhost:8000/")
    print("\nAvailable endpoints:")
    print("1. Generate an image:")
    print('curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d \'{"prompt":"a photo of a cat"}\'')
    print("\n2. Check job status:")
    print('curl "http://localhost:8000/status/<job_id>"')
    print("\n3. Check queue information:")
    print('curl "http://localhost:8000/queue-info"')
    print("\nPress Ctrl+C to shutdown gracefully")
    

except Exception as e:
    print(f"Error starting server: {e}")
    serve.shutdown()
    ray.shutdown()
    sys.exit(1)