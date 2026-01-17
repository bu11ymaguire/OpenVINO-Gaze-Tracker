import torch
from train import get_model

def export_to_onnx():
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("tuning/output/best_model.pth"))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 300, 300)
    output_path = "tuning/output/model.onnx"
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print("Export Complete!")

if __name__ == "__main__":
    export_to_onnx()
