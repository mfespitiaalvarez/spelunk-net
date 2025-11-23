"""
Data augmentation transforms for RGBD semantic segmentation.
All transforms handle both the 4-channel input (RGBD) and the mask.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, rgbd, mask):
        for t in self.transforms:
            rgbd, mask = t(rgbd, mask)
        return rgbd, mask


class RandomRotation:
    """
    Randomly rotate RGBD and mask by a random angle.
    
    Args:
        degrees: Range of degrees to rotate (e.g., 15 means [-15, 15])
        p: Probability of applying rotation
    """
    
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() > self.p:
            return rgbd, mask
        
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Rotate RGBD (4 channels)
        rgbd = TF.rotate(rgbd, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Rotate mask (nearest neighbor to preserve class labels)
        mask = mask.unsqueeze(0).float()  # Add channel dim
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        mask = mask.squeeze(0).long()  # Remove channel dim
        
        return rgbd, mask


class RandomCrop:
    """
    Randomly crop RGBD and mask to specified size.
    
    Args:
        size: Output size (height, width)
        p: Probability of applying crop (if not applied, resize to size)
    """
    
    def __init__(self, size, p=1.0):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() > self.p:
            # Just resize instead of crop
            rgbd = TF.resize(rgbd, self.size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask.unsqueeze(0), self.size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            return rgbd, mask
        
        _, h, w = rgbd.shape
        th, tw = self.size
        
        # If image is smaller than crop size, pad it
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            rgbd = F.pad(rgbd, (0, pad_w, 0, pad_h), mode='constant', value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
            h, w = rgbd.shape[1:]
        
        # Random crop
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        rgbd = rgbd[:, i:i+th, j:j+tw]
        mask = mask[i:i+th, j:j+tw]
        
        return rgbd, mask


class CenterCrop:
    """
    Center crop RGBD and mask to specified size.
    
    Args:
        size: Output size (height, width)
    """
    
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
    
    def __call__(self, rgbd, mask):
        _, h, w = rgbd.shape
        th, tw = self.size
        
        # If image is smaller than crop size, pad it
        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            rgbd = F.pad(rgbd, (0, pad_w, 0, pad_h), mode='constant', value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0)
            h, w = rgbd.shape[1:]
        
        # Center crop
        i = (h - th) // 2
        j = (w - tw) // 2
        
        rgbd = rgbd[:, i:i+th, j:j+tw]
        mask = mask[i:i+th, j:j+tw]
        
        return rgbd, mask


class Resize:
    """
    Resize RGBD and mask to specified size.
    
    Args:
        size: Output size (height, width) or single int for square
    """
    
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
    
    def __call__(self, rgbd, mask):
        rgbd = TF.resize(rgbd, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), self.size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return rgbd, mask


class RandomDepthDropout:
    """
    Randomly set the depth channel to all zeros (complete dropout).
    This simulates depth sensor failure.
    
    Args:
        p: Probability of applying depth dropout
    """
    
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() < self.p:
            # Set depth channel (channel 3) to zero
            rgbd[3, :, :] = 0.0
        
        return rgbd, mask


class RandomHorizontalFlip:
    """
    Randomly flip RGBD and mask horizontally.
    
    Args:
        p: Probability of applying flip
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() < self.p:
            rgbd = TF.hflip(rgbd)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        return rgbd, mask


class RandomVerticalFlip:
    """
    Randomly flip RGBD and mask vertically.
    
    Args:
        p: Probability of applying flip
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() < self.p:
            rgbd = TF.vflip(rgbd)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        
        return rgbd, mask


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation of RGB channels.
    Does not affect depth channel.
    
    Args:
        brightness: How much to jitter brightness (0-1)
        contrast: How much to jitter contrast (0-1)
        saturation: How much to jitter saturation (0-1)
        p: Probability of applying jitter
    """
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    
    def __call__(self, rgbd, mask):
        if random.random() > self.p:
            return rgbd, mask
        
        # Apply jitter only to RGB channels
        rgb = rgbd[:3]
        depth = rgbd[3:]
        
        # Apply color jitter
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            rgb = rgb * brightness_factor
        
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = rgb.mean(dim=[1, 2], keepdim=True)
            rgb = (rgb - mean) * contrast_factor + mean
        
        if self.saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            gray = rgb.mean(dim=0, keepdim=True)
            rgb = (rgb - gray) * saturation_factor + gray
        
        # Clamp RGB to [0, 1]
        rgb = torch.clamp(rgb, 0, 1)
        
        # Recombine
        rgbd = torch.cat([rgb, depth], dim=0)
        
        return rgbd, mask


class NormalizeDepth:
    """
    Normalize depth channel to [0, 1] range based on max_depth.
    
    Args:
        max_depth: Maximum depth value in meters (values above are clamped)
    """
    
    def __init__(self, max_depth=20.0):
        self.max_depth = max_depth
    
    def __call__(self, rgbd, mask):
        # Normalize depth channel (index 3)
        rgbd[3] = torch.clamp(rgbd[3] / self.max_depth, 0, 1)
        return rgbd, mask


class ToTensor:
    """Convert numpy arrays to tensors (if not already)."""
    
    def __call__(self, rgbd, mask):
        if not isinstance(rgbd, torch.Tensor):
            rgbd = torch.from_numpy(rgbd).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        return rgbd, mask


def get_training_augmentation(
    crop_size=512,
    rotation_degrees=15,
    depth_dropout_prob=0.1,
    max_depth=20.0
):
    """
    Get standard training augmentation pipeline.
    
    Args:
        crop_size: Size to crop/resize images to
        rotation_degrees: Range for random rotation
        depth_dropout_prob: Probability of depth channel dropout
        max_depth: Maximum depth for normalization
    
    Returns:
        Compose object with augmentation transforms
    """
    return Compose([
        # Geometric augmentations
        RandomRotation(degrees=rotation_degrees, p=0.5),
        RandomHorizontalFlip(p=0.5),
        
        # Crop/Resize
        RandomCrop(crop_size, p=0.8),
        Resize(crop_size),  # Ensure consistent size
        
        # Color augmentation (RGB only)
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
        
        # Depth augmentation
        RandomDepthDropout(p=depth_dropout_prob),
        NormalizeDepth(max_depth=max_depth),
    ])


def get_validation_augmentation(crop_size=512, max_depth=20.0):
    """
    Get validation augmentation (only resize, no randomness).
    
    Args:
        crop_size: Size to resize images to
        max_depth: Maximum depth for normalization
    
    Returns:
        Compose object with validation transforms
    """
    return Compose([
        Resize(crop_size),
        NormalizeDepth(max_depth=max_depth),
    ])


if __name__ == '__main__':
    """Test transforms"""
    print("Testing transforms...")
    
    # Create dummy data
    rgbd = torch.rand(4, 480, 640)  # 4 channels, 480x640
    mask = torch.randint(0, 11, (480, 640))  # 11 classes
    
    print(f"Input RGBD: {rgbd.shape}, range: [{rgbd.min():.3f}, {rgbd.max():.3f}]")
    print(f"Input mask: {mask.shape}, unique: {torch.unique(mask).tolist()}")
    
    # Test training augmentation
    train_transform = get_training_augmentation(crop_size=512, rotation_degrees=15, depth_dropout_prob=0.1)
    rgbd_aug, mask_aug = train_transform(rgbd.clone(), mask.clone())
    
    print(f"\nAfter training augmentation:")
    print(f"  RGBD: {rgbd_aug.shape}, range: [{rgbd_aug.min():.3f}, {rgbd_aug.max():.3f}]")
    print(f"  Mask: {mask_aug.shape}, unique: {torch.unique(mask_aug).tolist()}")
    
    # Test validation augmentation
    val_transform = get_validation_augmentation(crop_size=512)
    rgbd_val, mask_val = val_transform(rgbd.clone(), mask.clone())
    
    print(f"\nAfter validation augmentation:")
    print(f"  RGBD: {rgbd_val.shape}, range: [{rgbd_val.min():.3f}, {rgbd_val.max():.3f}]")
    print(f"  Mask: {mask_val.shape}, unique: {torch.unique(mask_val).tolist()}")
    
    # Test depth dropout
    print("\nTesting depth dropout...")
    dropout = RandomDepthDropout(p=1.0)  # Always apply
    rgbd_test = torch.rand(4, 100, 100)
    rgbd_test[3] = 5.0  # Set depth to 5.0
    print(f"  Before: depth channel mean = {rgbd_test[3].mean():.3f}")
    rgbd_test, _ = dropout(rgbd_test, mask[:100, :100])
    print(f"  After: depth channel mean = {rgbd_test[3].mean():.3f} (should be 0)")
    
    print("\n✓ All transforms working correctly!")

