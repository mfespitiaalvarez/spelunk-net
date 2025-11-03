"""
Training script for Fast-SCNN with 4-channel input (RGB + Depth).

Usage (run from project root):
    python src/scripts/train_4ch.py --data_dir src/perception_sim/training_images

This script demonstrates how to:
1. Load your perception sim generated dataset
2. Initialize Fast-SCNN with 4 channels
3. Load pretrained 3-channel weights
4. Freeze RGB channels and train only the depth channel
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Add Fast-SCNN utils to path
sys.path.insert(0, os.path.join(project_root, 'external/Fast-SCNN-pytorch'))

from src.models.fast_scnn_4ch import (
    FastSCNN4Ch, 
    load_pretrained_with_new_channel,
    freeze_all_except_4th_channel_and_classifier
)
from src.datasets.cones_balls_dataset import ConesBallsDataset
from utils.loss import MixSoftmaxCrossEntropyLoss
from utils.metric import SegmentationMetric
from utils.lr_scheduler import LRScheduler


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, metric, device, cur_iter):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for rgbd, masks in pbar:
        rgbd = rgbd.to(device)
        masks = masks.to(device)
        
        # Update learning rate
        lr = lr_scheduler(cur_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(rgbd)
        
        # Calculate loss (criterion handles aux loss automatically)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        predictions = outputs[0].argmax(dim=1).cpu().numpy()
        masks_np = masks.cpu().numpy()
        metric.update(predictions, masks_np)
        
        pixAcc, mIoU = metric.get()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'pixAcc': pixAcc, 'mIoU': mIoU, 'lr': lr})
        
        cur_iter += 1
    
    return epoch_loss / len(dataloader), cur_iter


def validate(model, dataloader, criterion, metric, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for rgbd, masks in pbar:
            rgbd = rgbd.to(device)
            masks = masks.to(device)
            
            outputs = model(rgbd)
            
            # Calculate loss (criterion handles aux loss automatically)
            loss = criterion(outputs, masks)
            
            # Update metrics
            predictions = outputs[0].argmax(dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            metric.update(predictions, masks_np)
            
            pixAcc, mIoU = metric.get()
            val_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'pixAcc': pixAcc, 'mIoU': mIoU})
    
    return val_loss / len(dataloader)


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load label mapping to get number of classes
    label_map_path = os.path.join(args.data_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_info = json.load(f)
        # Count unique classes: background + cones + balls
        num_classes = 1 + len(label_info['cones']) + len(label_info['balls'])
        print(f"Number of classes: {num_classes}")
        print(f"  Background: 1")
        print(f"  Cones: {len(label_info['cones'])} ({list(label_info['cones'].keys())})")
        print(f"  Balls: {len(label_info['balls'])} ({list(label_info['balls'].keys())})")
    else:
        print(f"Warning: Label mapping not found at {label_map_path}")
        num_classes = 11  # Default: 0 (background) + 6 cones + 4 balls
    
    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}")
    full_dataset = ConesBallsDataset(
        args.data_dir,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"\nCreating 4-channel Fast-SCNN model...")
    model = FastSCNN4Ch(num_classes=num_classes, in_channels=4, aux=args.aux)
    
    # Load pretrained weights if available
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading weights from {args.pretrained_path}")
        
        # Check if this is a 3-channel Cityscapes model or our own 4-channel checkpoint
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        first_conv_shape = checkpoint.get('learning_to_downsample.conv.conv.0.weight').shape
        is_3channel = first_conv_shape[1] == 3  # Check input channels
        
        if is_3channel:
            print("→ Detected 3-channel pretrained model (e.g., Cityscapes)")
            print(f"  Converting to 4-channel with init_method={args.init_method}")
            # Load pretrained weights WITHOUT freezing first
            model = load_pretrained_with_new_channel(
                model,
                args.pretrained_path,
                freeze_rgb=False,  # Don't freeze yet
                init_method=args.init_method
            )
        else:
            print("→ Detected 4-channel checkpoint (our own model)")
            # Direct load - already 4-channel
            model.load_state_dict(checkpoint, strict=True)
            print("✓ Checkpoint loaded successfully")
        
        # Now apply freezing strategy based on args
        if args.unfreeze_all:
            print("✓ All parameters trainable (Stage 2: Fine-tuning)")
        elif args.freeze_rgb:
            print("\nApplying freeze strategy: EVERYTHING except 4th channel and classifier head (Stage 1: Transfer Learning)")
            freeze_all_except_4th_channel_and_classifier(model)
        else:
            print("✓ All parameters trainable (default)")
    else:
        if args.pretrained_path:
            print(f"Warning: Pretrained weights not found at {args.pretrained_path}")
        print("Training from scratch (no pretrained weights)...")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function (handles auxiliary loss automatically)
    criterion = MixSoftmaxCrossEntropyLoss(aux=args.aux, aux_weight=0.4, ignore_label=-1)
    
    # Metrics
    train_metric = SegmentationMetric(num_classes)
    val_metric = SegmentationMetric(num_classes)
    
    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (poly schedule common for segmentation)
    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    lr_scheduler = LRScheduler(
        mode=args.lr_scheduler,
        base_lr=args.lr,
        nepochs=args.epochs,
        iters_per_epoch=iters_per_epoch,
        power=0.9
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_val_mIoU = 0.0
    cur_iter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Reset metrics
        train_metric.reset()
        val_metric.reset()
        
        # Train
        train_loss, cur_iter = train_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler, train_metric, device, cur_iter
        )
        train_pixAcc, train_mIoU = train_metric.get()
        print(f"Train Loss: {train_loss:.4f}, pixAcc: {train_pixAcc:.4f}, mIoU: {train_mIoU:.4f}")
        
        # Validate
        if len(val_dataset) > 0:
            val_loss = validate(model, val_loader, criterion, val_metric, device)
            val_pixAcc, val_mIoU = val_metric.get()
            print(f"Val Loss: {val_loss:.4f}, pixAcc: {val_pixAcc:.4f}, mIoU: {val_mIoU:.4f}")
            
            # Save best model based on mIoU
            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✓ Saved best model to {checkpoint_path} (mIoU: {val_mIoU:.4f})")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mIoU': train_mIoU,
                'val_mIoU': val_mIoU if len(val_dataset) > 0 else None,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model saved to {final_path}")
    if len(val_dataset) > 0:
        print(f"Best validation mIoU: {best_val_mIoU:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fast-SCNN with 4-channel input')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='src/perception_sim/training_images',
                        help='Path to training data directory (from project root)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--depth_scale', type=float, default=1.0,
                        help='Depth scaling factor')
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help='Maximum depth for normalization')
    
    # Model
    parser.add_argument('--pretrained_path', type=str, default='external/Fast-SCNN-pytorch/weights/fast_scnn_citys.pth',
                        help='Path to pretrained weights or checkpoint')
    parser.add_argument('--freeze_rgb', action='store_true',
                        help='Freeze EVERYTHING except 4th channel and classifier head (Stage 1: transfer learning)')
    parser.add_argument('--unfreeze_all', action='store_true',
                        help='Train all parameters (Stage 2: fine-tuning). Overrides --freeze_rgb')
    parser.add_argument('--init_method', type=str, default='random',
                        choices=['random', 'zeros', 'mean', 'copy_r', 'copy_g', 'copy_b'],
                        help='Initialization method for 4th channel')
    parser.add_argument('--aux', action='store_true', default=True,
                        help='Use auxiliary loss')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.045,
                        help='Base learning rate (default 0.045 for SGD)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'cosine', 'step', 'linear', 'constant'],
                        help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=4e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (from project root)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    main(args)

