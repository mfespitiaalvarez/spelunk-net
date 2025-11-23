"""
Convert PyTorch model to ONNX with label remapping documented.

Since the model was trained with CVAT's alphabetical label ordering,
but deployment expects semantic_labels_params.yaml ordering,
you need to apply remapping AFTER inference.

For C++ deployment, use the mapping defined in label_mapping.py
"""

import torch
import sys
import os
import argparse

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external/Fast-SCNN-pytorch'))

from src.models.fast_scnn_4ch import FastSCNN4Ch

# Import the remapping
sys.path.insert(0, project_root)
from label_mapping import TRAINED_TO_DEPLOYMENT


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX (with remapping note)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: same as checkpoint with .onnx)')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of output classes')
    parser.add_argument('--height', type=int, default=512,
                        help='Input height')
    parser.add_argument('--width', type=int, default=512,
                        help='Input width')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    args = parser.parse_args()
    
    # Set output path
    checkpoint_path = args.checkpoint
    if args.output:
        onnx_path = args.output
    else:
        onnx_path = checkpoint_path.replace('.pth', '.onnx')
    
    num_classes = args.num_classes
    batch_size = 1
    height = args.height
    width = args.width

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Input size: {height}x{width}")
    
    model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  - Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded")

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
    print(f"\nExporting to ONNX...")
    print(f"  - Output: {onnx_path}")
    print(f"  - Opset version: {args.opset}")
    
    dummy_input = torch.randn(batch_size, 4, height, width)

    # Export to ONNX
    torch.onnx.export(
        deployment_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"\n✓ Model exported successfully!")
    print(f"  - ONNX file: {onnx_path}")
    print(f"  - Input shape: (batch, 4, {height}, {width})")
    print(f"  - Output shape: (batch, {num_classes}, {height}, {width})")
    
    # Get file size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  - File size: {size_mb:.2f} MB")
    
    print(f"\n{'='*70}")
    print("⚠️  IMPORTANT: LABEL REMAPPING REQUIRED!")
    print(f"{'='*70}")
    print("This model was trained with CVAT's alphabetical label ordering,")
    print("which does NOT match semantic_labels_params.yaml.")
    print()
    print("After inference, you MUST apply label remapping:")
    print()
    print("In C++, use this mapping:")
    print("  int remapping[11] = {0, 9, 6, 10, 4, 2, 3, 7, 1, 8, 5};")
    print("  remapped_class_id = remapping[predicted_class_id];")
    print()
    print("Or see label_mapping.py for Python implementation.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

