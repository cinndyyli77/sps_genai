
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

import torch
from torchvision import transforms

from helper_lib.model import CNN
from app.main import app as original_router


# Create FastAPI application
app = FastAPI(title="CNN Image Classifier")

# Include the original Assignment 1 API
app.include_router(original_router)


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Load trained CNN model
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = CNN().to(device)

checkpoint = torch.load(
    "./cnn_checkpoint.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# CNN API root
@app.get("/")
def root():
    return {
        "message": "CNN Image Classifier API is running"
    }


# CNN image prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(
            outputs, dim=1
        ).item()

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class]
    }
