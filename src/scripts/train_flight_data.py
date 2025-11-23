"""
Training script for Fast-SCNN with real flight data (RGBD + CVAT annotations).

Usage (run from project root):
    python src/scripts/train_flight_data.py --data_root training_data --epochs 100

Features:
- Loads real flight RGBD data with CVAT annotations
- Applies augmentation (rotation, crop, depth dropout)
- Handles missing annotations (assumes background)
- Supports transfer learning from Cityscapes pretrained model
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
from src.datasets.flight_dataset import FlightDataset
from src.datasets.transforms import get_training_augmentation, get_validation_augmentation
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
    
    # Create augmentation transforms
    print(f"\nSetting up augmentation:")
    print(f"  - Crop size: {args.crop_size}")
    print(f"  - Rotation: ±{args.rotation_degrees}°")
    print(f"  - Depth dropout probability: {args.depth_dropout_prob}")
    print(f"  - Max depth: {args.max_depth}m")
    
    train_transform = get_training_augmentation(
        crop_size=args.crop_size,
        rotation_degrees=args.rotation_degrees,
        depth_dropout_prob=args.depth_dropout_prob,
        max_depth=args.max_depth
    )
    
    val_transform = get_validation_augmentation(
        crop_size=args.crop_size,
        max_depth=args.max_depth
    )
    
    # Create datasets
    print(f"\nLoading flight dataset from {args.data_root}")
    
    # Parse session folders
    if args.session_folders:
        session_folders = args.session_folders
    else:
        # Backward compatibility: use single session_folder if provided
        session_folders = [args.session_folder] if args.session_folder else None
    
    # Parse frame selection
    frame_selection = None
    if args.frame_selection:
        if isinstance(args.frame_selection, str):
            # From command line as JSON string
            frame_selection = json.loads(args.frame_selection)
        elif isinstance(args.frame_selection, dict):
            # Already a dict (from config file)
            frame_selection = args.frame_selection
        else:
            frame_selection = None
    
    # Full dataset with training augmentation
    train_dataset = FlightDataset(
        args.data_root,
        session_folder=session_folders,
        frame_selection=frame_selection,
        transform=train_transform,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )
    
    # Validation dataset with minimal transforms
    val_dataset = FlightDataset(
        args.data_root,
        session_folder=session_folders,
        frame_selection=frame_selection,
        transform=val_transform,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth
    )
    
    num_classes = train_dataset.num_classes
    class_names = train_dataset.get_class_names()
    print(f"\nDataset info:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    
    # Split into train/val with random shuffling (for temporal data)
    total_size = len(train_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    # Randomize indices to avoid temporal bias (but keep reproducible with seed)
    import random
    indices = list(range(total_size))
    random.seed(42)  # Fixed seed for reproducibility across runs
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Save train/val indices for reproducibility in inference
    split_info = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'total_size': total_size,
        'val_split': args.val_split,
        'session_folders': session_folders,
        'frame_selection': frame_selection
    }
    split_info_path = os.path.join(args.output_dir, 'train_val_split.json')
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ Train/val split saved to {split_info_path}")
    
    print(f"  - Total samples: {total_size}")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Split: Randomized (seed=42) to avoid temporal bias")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False  # Use all training samples (small dataset)
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
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        first_conv_key = 'learning_to_downsample.conv.conv.0.weight'
        if first_conv_key in state_dict:
            first_conv_shape = state_dict[first_conv_key].shape
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
                model.load_state_dict(state_dict, strict=True)
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
                'num_classes': num_classes,
                'class_names': class_names,
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


def load_config_from_json(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


if __name__ == '__main__':
    # First, parse just the config argument
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=None,
                               help='Path to JSON config file')
    config_args, remaining = config_parser.parse_known_args()
    
    # Load config from file if provided
    config_dict = {}
    if config_args.config:
        config_dict = load_config_from_json(config_args.config)
    
    parser = argparse.ArgumentParser(description='Train Fast-SCNN with real flight RGBD data')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file. Command-line arguments override config file values.')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                        default=config_dict.get('data_root', 'training_data'),
                        help='Root directory containing flight_original_imgs and flight_cvat_annotations')
    parser.add_argument('--session_folder', type=str, 
                        default=config_dict.get('session_folder', 'flight'),
                        help='Name of the task folder (e.g., "flight", "hand_medium") - backward compatibility')
    parser.add_argument('--session_folders', type=str, nargs='+', 
                        default=config_dict.get('session_folders', None),
                        help='List of task folder names (e.g., "flight", "hand_medium"). '
                             'Each folder should have imgs/color, imgs/depth, and annotations/ subdirectories. '
                             'Example: --session_folders flight hand_medium')
    parser.add_argument('--frame_selection', type=str, default=None,
                        help='JSON string mapping task folder names to frame lists or -1 for all. '
                             'Example: \'{"flight": ["000026", "000027"], "hand_medium": -1}\' '
                             'Use -1 to use all frames from a task folder.')
    parser.add_argument('--val_split', type=float, 
                        default=config_dict.get('val_split', 0.2),
                        help='Validation split ratio')
    parser.add_argument('--depth_scale', type=float, 
                        default=config_dict.get('depth_scale', 1.0),
                        help='Depth scaling factor')
    parser.add_argument('--max_depth', type=float, 
                        default=config_dict.get('max_depth', 20.0),
                        help='Maximum depth for normalization (meters)')
    
    # Augmentation
    parser.add_argument('--crop_size', type=int, 
                        default=config_dict.get('crop_size', 512),
                        help='Crop size for augmentation')
    parser.add_argument('--rotation_degrees', type=int, 
                        default=config_dict.get('rotation_degrees', 15),
                        help='Max rotation angle for augmentation')
    parser.add_argument('--depth_dropout_prob', type=float, 
                        default=config_dict.get('depth_dropout_prob', 0.1),
                        help='Probability of complete depth channel dropout')
    
    # Model
    parser.add_argument('--pretrained_path', type=str, 
                        default=config_dict.get('pretrained_path', 'external/Fast-SCNN-pytorch/weights/fast_scnn_citys.pth'),
                        help='Path to pretrained weights or checkpoint')
    parser.add_argument('--freeze_rgb', action='store_true',
                        default=config_dict.get('freeze_rgb', False),
                        help='Freeze EVERYTHING except 4th channel and classifier head (Stage 1: transfer learning)')
    parser.add_argument('--unfreeze_all', action='store_true',
                        default=config_dict.get('unfreeze_all', False),
                        help='Train all parameters (Stage 2: fine-tuning). Overrides --freeze_rgb')
    parser.add_argument('--init_method', type=str, 
                        default=config_dict.get('init_method', 'random'),
                        choices=['random', 'zeros', 'mean', 'copy_r', 'copy_g', 'copy_b'],
                        help='Initialization method for 4th channel')
    parser.add_argument('--aux', action='store_true', 
                        default=config_dict.get('aux', True),
                        help='Use auxiliary loss')
    
    # Training
    parser.add_argument('--epochs', type=int, 
                        default=config_dict.get('epochs', 100),
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, 
                        default=config_dict.get('batch_size', 8),
                        help='Batch size')
    parser.add_argument('--lr', type=float, 
                        default=config_dict.get('lr', 0.045),
                        help='Base learning rate (default 0.045 for SGD)')
    parser.add_argument('--lr_scheduler', type=str, 
                        default=config_dict.get('lr_scheduler', 'poly'),
                        choices=['poly', 'cosine', 'step', 'linear', 'constant'],
                        help='Learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, 
                        default=config_dict.get('weight_decay', 4e-5),
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, 
                        default=config_dict.get('num_workers', 4),
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, 
                        default=config_dict.get('output_dir', 'checkpoints/flight_training'),
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, 
                        default=config_dict.get('save_interval', 10),
                        help='Save checkpoint every N epochs')
    
    # Parse arguments (command-line will override config defaults)
    args = parser.parse_args(remaining)
    
    # Handle frame_selection from config file (if not provided via CLI)
    if not args.frame_selection and 'frame_selection' in config_dict:
        # Store the dict directly in args (will be handled in main())
        args.frame_selection = config_dict['frame_selection']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save final config (merged from file + command-line)
    # Convert frame_selection to JSON string for saving
    config_to_save = vars(args).copy()
    if isinstance(config_to_save.get('frame_selection'), dict):
        config_to_save['frame_selection'] = json.dumps(config_to_save['frame_selection'])
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    main(args)

