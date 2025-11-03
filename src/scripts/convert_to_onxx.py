# convert_to_onnx.py
import torch
import sys
import os

# Add paths
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external/Fast-SCNN-pytorch'))

from src.models.fast_scnn_4ch import FastSCNN4Ch

# Configuration
checkpoint_path = "checkpoints/from_scratch/best_model.pth"
onnx_path = "checkpoints/from_scratch/best_model.onnx"
num_classes = 11
batch_size = 1
height = 640  # Adjust to your input size
width = 480   # Adjust to your input size

# Load model
print("Loading model...")
model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=True)
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Wrap model to return only main output
class DeploymentModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs[0]  # Return only main output, ignore aux

deployment_model = DeploymentModel(model)
deployment_model.eval()

# Create dummy input (4 channels: RGB + Depth)
dummy_input = torch.randn(batch_size, 4, height, width)

# Export to ONNX
print(f"Exporting to ONNX: {onnx_path}")
torch.onnx.export(
    deployment_model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)

print(f"✓ Model exported to {onnx_path}")

# Optional: Verify the export
# import onnx
# onnx_model = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model)
# print("✓ ONNX model verified!")