"""
PyTorch Dataset for real flight data with CVAT annotations.
Handles RGBD input with augmentation for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
from pathlib import Path


class FlightDataset(Dataset):
    """
    Dataset for real flight data with CVAT segmentation annotations.
    
    Directory structure expected:
        data_root/
            <task>/  (e.g., "flight", "hand_medium")
                imgs/
                    color/
                        frame_000026_1760646416223764315.png
                        ...
                    depth/
                        frame_000026_1760646416223764316.png
                        ...
                annotations/
                    SegmentationClass/
                        frame_000026_1760646416223764315.png
                        ...
                    labelmap.txt
    
    Returns:
        rgbd: (4, H, W) tensor - RGB + Depth combined
        mask: (H, W) tensor - integer labels 0-10
    """
    
    def __init__(
        self,
        data_root,
        session_folder="10_16_different_light_flight_semantics_hard_sampling_20251027_123252",
        frame_selection=None,
        transform=None,
        depth_scale=1.0,
        max_depth=20.0,
    ):
        """
        Args:
            data_root: Root directory (e.g., 'training_data/')
            session_folder: Name of the task folder (str) or list of task folder names
                           (e.g., "flight", "hand_medium", or ["flight", "hand_medium"])
            frame_selection: Dict mapping task folder name -> list of frame numbers (as strings like "000026") 
                            or -1 for all frames. If None, uses all frames from all folders.
                            Example: {"flight": ["000026", "000027"], "hand_medium": -1}
            transform: Augmentation transforms to apply
            depth_scale: Scale factor for depth values
            max_depth: Maximum depth value for normalization
        """
        self.data_root = Path(data_root)
        
        # Handle backward compatibility: convert single folder to list
        if isinstance(session_folder, str):
            self.task_folders = [session_folder]
        else:
            self.task_folders = session_folder
        
        self.frame_selection = frame_selection or {}
        self.transform = transform
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        
        # Find all color images and match with depth/masks from all task folders
        self.samples = self._build_sample_list()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in any of the specified task folders: {self.task_folders}")
        
        # Load label mapping from first task (assuming all tasks use same labelmap)
        # Try to find labelmap in any of the task folders
        self.label_map = None
        self.labelmap_path = None
        for task_folder in self.task_folders:
            labelmap_path = self.data_root / task_folder / "annotations" / "labelmap.txt"
            if labelmap_path.exists():
                self.labelmap_path = labelmap_path
                break
        
        if self.labelmap_path is None:
            raise ValueError(f"Labelmap not found in any task folder: {self.task_folders}")
        
        self.label_map = self._parse_labelmap()
        self.num_classes = len(self.label_map)
        
        print(f"FlightDataset initialized:")
        print(f"  - {len(self.samples)} samples")
        print(f"  - {self.num_classes} classes")
        print(f"  - Task folders: {self.task_folders}")
        if self.frame_selection:
            print(f"  - Frame selection: {self.frame_selection}")
    
    def _parse_labelmap(self):
        """
        Parse CVAT labelmap.txt format:
        # label:color_rgb:parts:actions
        background:0,0,0::
        blue_ball:106,106,243::
        ...
        
        Returns dict: {label_name: (class_id, (r, g, b))}
        """
        label_map = {}
        class_id = 0  # Sequential class ID counter
        with open(self.labelmap_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(':')
                if len(parts) >= 2:
                    label_name = parts[0]
                    color_str = parts[1]
                    rgb = tuple(map(int, color_str.split(',')))
                    label_map[label_name] = (class_id, rgb)
                    class_id += 1  # Increment for next class
        
        return label_map
    
    def _build_sample_list(self):
        """
        Build list of samples by scanning color directories from all task folders.
        Each sample includes frame_id, task folder, and annotation info.
        """
        samples = []
        
        # Process each task folder
        for task_folder in self.task_folders:
            color_dir = self.data_root / task_folder / "imgs" / "color"
            depth_dir = self.data_root / task_folder / "imgs" / "depth"
            mask_dir = self.data_root / task_folder / "annotations" / "SegmentationClass"
            
            # Validate directories exist
            if not color_dir.exists():
                print(f"Warning: Color directory not found: {color_dir}, skipping...")
                continue
            if not depth_dir.exists():
                print(f"Warning: Depth directory not found: {depth_dir}, skipping...")
                continue
            if not mask_dir.exists():
                print(f"Warning: Mask directory not found: {mask_dir}, skipping...")
                continue
            
            # Get all mask files for this task (for quick lookup)
            mask_files = {f.name for f in mask_dir.glob("frame_*.png")}
            
            # Get frame selection for this task folder
            folder_frames = self.frame_selection.get(task_folder, -1)
            
            # Get all color images
            color_files = sorted(color_dir.glob("frame_*.png"))
            
            for color_path in color_files:
                # Extract frame info from filename
                # Format: frame_000026_1760646416223764315.png
                frame_name = color_path.stem  # Without .png
                
                # Check if corresponding depth exists
                # Depth might have slightly different timestamp
                frame_num_match = re.match(r'frame_(\d+)_(\d+)', frame_name)
                if not frame_num_match:
                    continue
                
                frame_num = frame_num_match.group(1)
                
                # Filter by frame selection if specified
                if folder_frames != -1:
                    # folder_frames is a list of frame numbers (as strings or ints)
                    # Normalize frame numbers: remove leading zeros for comparison
                    frame_num_normalized = str(int(frame_num))
                    folder_frames_normalized = [str(int(f)) if isinstance(f, (int, str)) and str(f).isdigit() else str(f) for f in folder_frames]
                    if frame_num_normalized not in folder_frames_normalized:
                        continue
                
                # Find matching depth file (same frame number, possibly different timestamp)
                depth_files = list(depth_dir.glob(f"frame_{frame_num}_*.png"))
                if not depth_files:
                    print(f"Warning: No depth file for {frame_name} in {task_folder}")
                    continue
                
                depth_path = depth_files[0]  # Take first match
                
                # Check if annotation exists
                # Look for exact match or same frame number
                has_annotation = False
                mask_path = None
                
                # First try exact match
                if f"{frame_name}.png" in mask_files:
                    mask_path = mask_dir / f"{frame_name}.png"
                    has_annotation = True
                else:
                    # Try to find mask with same frame number
                    mask_candidates = list(mask_dir.glob(f"frame_{frame_num}_*.png"))
                    if mask_candidates:
                        mask_path = mask_candidates[0]
                        has_annotation = True
                
                samples.append({
                    'frame_name': frame_name,
                    'frame_num': frame_num,
                    'task_folder': task_folder,
                    'color_path': color_path,
                    'depth_path': depth_path,
                    'mask_path': mask_path,
                    'has_annotation': has_annotation
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_mask_from_rgb(self, mask_path):
        """
        Convert RGB mask to class indices using labelmap.
        """
        mask_rgb = np.array(Image.open(mask_path))
        if mask_rgb.ndim == 2:
            # Already grayscale
            return mask_rgb
        
        # Convert RGB to class indices
        h, w = mask_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        
        for label_name, (class_id, rgb) in self.label_map.items():
            # Find pixels matching this color
            match = np.all(mask_rgb[:, :, :3] == rgb, axis=2)
            mask[match] = class_id
        
        return mask
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB
        rgb = np.array(Image.open(sample['color_path']))
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        
        # Load depth (16-bit PNG in millimeters)
        depth = np.array(Image.open(sample['depth_path'])).astype(np.float32)
        # Check if depth is 16-bit (typical for depth sensors)
        if depth.max() > 255:
            depth = depth / 1000.0  # Convert mm to meters
        
        # Load or create mask
        if sample['has_annotation']:
            mask = self._load_mask_from_rgb(sample['mask_path'])
        else:
            # All background
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.int64)
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0).float() * self.depth_scale  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        # Combine RGBD
        rgbd = torch.cat([rgb, depth], dim=0)  # (4, H, W)
        
        # Apply transforms (handles augmentation)
        if self.transform:
            rgbd, mask = self.transform(rgbd, mask)
        
        return rgbd, mask
    
    def get_class_names(self):
        """Return ordered list of class names."""
        names = [''] * len(self.label_map)
        for name, (idx, _) in self.label_map.items():
            names[idx] = name
        return names


if __name__ == '__main__':
    import sys
    
    # Test the dataset
    data_root = sys.argv[1] if len(sys.argv) > 1 else "training_data"
    
    print(f"Testing FlightDataset with data_root: {data_root}")
    
    dataset = FlightDataset(data_root)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Classes: {dataset.get_class_names()}")
    
    # Load a sample
    rgbd, mask = dataset[0]
    print(f"\nSample 0:")
    print(f"  RGBD shape: {rgbd.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  RGB range: [{rgbd[:3].min():.3f}, {rgbd[:3].max():.3f}]")
    print(f"  Depth range: [{rgbd[3].min():.3f}, {rgbd[3].max():.3f}] meters")
    print(f"  Unique labels: {torch.unique(mask).tolist()}")
    print(f"  Label counts: {[(dataset.get_class_names()[l], (mask == l).sum().item()) for l in torch.unique(mask)]}")
    
    # Test batch loading
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    rgbd_batch, mask_batch = next(iter(dataloader))
    print(f"\nBatch test:")
    print(f"  RGBD batch shape: {rgbd_batch.shape}")
    print(f"  Mask batch shape: {mask_batch.shape}")
    print(f"✓ Dataset working correctly!")

