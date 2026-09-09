import os
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
EMOJIS = {'buildings': '🏙️', 'forest': '🌲', 'glacier': '🏔️',
          'mountain': '⛰️', 'sea': '🌊', 'street': '🛣️'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('models/resnet_scene.pth', map_location='cpu'))
model.eval()

def classify(image):
    img = Image.fromarray(image).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return {f"{EMOJIS[CLASSES[i]]} {CLASSES[i]}": float(probs[i]) for i in range(6)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=6),
    title="🌍 SceneVision — Scene Classifier",
    description="Upload an image and ResNet50 will classify the scene into one of 6 categories.",
    examples=[],
    theme=gr.themes.Ocean()
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))