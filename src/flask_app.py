from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from diffuser import create_pipe, run_pipe

app = Flask(__name__)
pipe = create_pipe()

# Pydantic model to validate request body
class Item(BaseModel):
    prompt: str

@app.route("/", methods=["GET"])
def read_root():
    return {"Hello": "World"}

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Validate and parse the JSON body using Pydantic
        item = Item(**request.json)
        print(item.prompt)
        # Call the diffuser pipeline function
        images = run_pipe(pipe, item.prompt)
        return jsonify({"status": "success"}), 200
    except ValidationError as e:
        # Handle validation error
        return jsonify(e.errors()), 422

if __name__ == "__main__":
    app.run(host="0.0.0.0" ,port=8080, debug=True)
