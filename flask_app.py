from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io

app = FastAPI()

# Load the saved Resnet model
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load('resnet_brain_tumor_model.pth'))
resnet_model.eval()

# Define the image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = resnet_model(image)
        _, predicted = torch.max(output, 1)

    return {"prediction": "Tumor" if predicted.item() == 1 else "No Tumor"}

# To run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
