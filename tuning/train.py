import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import os

def get_model(num_classes):
    # Load pre-trained SSD model
    print("Loading SSD300 VGG16 model...")
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    
    # Replace the head for our number of classes (Background + Face = 2 classes)
    in_channels = [672, 600, 512, 256, 256, 256] # VGG16 output channels
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")
    
    # 2 Classes: 0=Background, 1=Face
    model = get_model(num_classes=2)
    model.to(device)
    
    # Dummy Optimizer & Training Loop (Placeholder)
    # 실제 데이터셋 로더(Dataset Class) 구현 필요
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    print("Model is ready for training!")
    print("To proceed, we need real data in 'tuning/data'.")
    
    # Save dummy weights for export testing
    os.makedirs("tuning/output", exist_ok=True)
    torch.save(model.state_dict(), "tuning/output/best_model.pth")
    print("Saved skeletal model to tuning/output/best_model.pth")

if __name__ == "__main__":
    train()
