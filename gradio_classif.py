import gradio as gr
import torch
from torchvision import transforms
import requests 
from PIL import Image

model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.eval()

response = requests.get("https://git.io/JJkYN", verify=False)
labels = response.text.split("\n")

def predict(inp):
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0) 
    return {labels[i]: float(prediction[i]) for i in range(1000)}

inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=5)
interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)

interface.launch()