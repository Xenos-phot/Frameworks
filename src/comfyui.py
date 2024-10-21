import json
from time import time
import os
from PIL import Image
import io

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse


with open('workflow_api.json','r', encoding='utf-8') as f:
    workflow_data = f.read()

output_dir = '../output'

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):

    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    queue_prompt_time =time()
    prompt_id = queue_prompt(prompt)['prompt_id']
    print(f'time taken to queue prompt:',time()-queue_prompt_time)

    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

wf = json.loads(workflow_data)
with open('../prompts.json', 'r') as f:
    prompts = json.load(f)

average_time =0
#set the seed for our KSampler node
wf["3"]["inputs"]["seed"] = 12
# set num of images
wf["5"]["inputs"]["batch_size"] = 4
for prompt_index in range(len(prompts)):
    prompt= prompts[f"{prompt_index}"]
    
    # Create a directory for each prompt
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index}")
    os.makedirs(prompt_dir, exist_ok=True)

    #set the text prompt for our positive CLIPTextEncode
    wf["6"]["inputs"]["text"] = prompt

    

    start_time  =time()
    images = get_images(ws, wf)
    print(f'Time taken for prompt {prompt_index} is {time()-start_time}')
    average_time +=time()-start_time
    for i,node_id in enumerate(images):
        for j, image_data in enumerate(images[node_id]):
            
            image = Image.open(io.BytesIO(image_data))
            image.save(os.path.join(prompt_dir,f'comfy_{j}.png'))

print(f'Average time taken: {average_time/len(prompts)}')
ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts

