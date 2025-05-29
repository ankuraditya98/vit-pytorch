from PIL import Image
from torchvision import transforms
import timm
import torch
import requests

# print(timm.models.get_model_weights('vit_base_patch16_224'))

# Step 1: Load a pre-trained Vision Transformer model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()  # Set the model to evaluation mode

# print("Model loaded:", model)


# Step 2: Preprocess the data
# Load your image
image_path = "dog.jpg"  # <-- Replace with your image path
img = Image.open(image_path).convert("RGB")

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # Resize to model's input
    transforms.ToTensor(),                          # Convert to tensor [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3)           # Normalize to [-1, 1]
])

# Apply preprocessing
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

print("Image tensor shape:", img_tensor.shape)  # Should be [1, 3, 224, 224]



# Step 4: Run inference
with torch.no_grad():  # No gradient computation
    outputs = model(img_tensor)  # Shape: [1, 1000]
    predicted_class_idx = outputs.argmax(dim=1).item()

print(f"Predicted class index: {predicted_class_idx}")
# print(f"Raw model output: {outputs}")




# Step 5: Download ImageNet class labels
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(labels_url).text.strip().split("\n")

# Print human-readable label
predicted_label = imagenet_classes[predicted_class_idx]
print(f"Predicted label: {predicted_label}")