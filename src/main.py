from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from diffuser import create_pipe, run_pipe

app = FastAPI()
pipe =create_pipe()


class Item(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate(item: Item):
    print(item.prompt)
    images = run_pipe(pipe, item.prompt)
    return 200