"""
Extract color and depth image pairs from MCAP ROS bags with multiple sampling strategies.

Supports three sampling modes:
- random: Random sampling across entire dataset (default)
- gaussian: Gaussian-weighted sampling around target timestamps
- hard: Gaussian sampling with minimum time gap enforcement to avoid temporal redundancy
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def find_project_root(start_path: Path = None) -> Path:
    """
    Find the project root directory by looking for the spelunk-net directory.
    
    Args:
        start_path: Starting path to search from (defaults to script location)
        
    Returns:
        Path to the project root directory
    """
    if start_path is None:
        start_path = Path(__file__).resolve()
    
    current = start_path if start_path.is_dir() else start_path.parent
    
    while current.parent != current:  # Stop at filesystem root
        if current.name == 'spelunk-net':
            return current
        current = current.parent
    
    # Fallback: if we can't find spelunk-net, assume we're in src/ and go up one level
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON config file
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def decode_image(msg) -> np.ndarray:
    """
    Decode a ROS2 Image message to a numpy array.
    
    Args:
        msg: ROS2 sensor_msgs/msg/Image message
        
    Returns:
        numpy array of the image
    """
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    data = bytes(msg.data)  # Convert to bytes if needed
    
    # Convert bytes to numpy array based on encoding
    if encoding == "mono8" or encoding == "8UC1":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
    elif encoding == "mono16" or encoding == "16UC1":
        img = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
    elif encoding == "bgr8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    elif encoding == "rgb8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "bgra8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    elif encoding == "rgba8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    elif encoding == "32FC1":
        img = np.frombuffer(data, dtype=np.float32).reshape(height, width)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    
    return img


def find_matching_pairs(color_msgs: List[Tuple], depth_msgs: List[Tuple], 
                       max_time_diff_ns: int = 50_000_000) -> List[Tuple[int, int]]:
    """
    Find matching color and depth message pairs based on timestamp proximity.
    
    Args:
        color_msgs: List of (timestamp, message) tuples for color images
        depth_msgs: List of (timestamp, message) tuples for depth images
        max_time_diff_ns: Maximum time difference in nanoseconds (default: 50ms)
        
    Returns:
        List of (color_idx, depth_idx) pairs
    """
    pairs = []
    depth_idx = 0
    
    for color_idx, (color_ts, _) in enumerate(color_msgs):
        # Find the closest depth message
        while depth_idx < len(depth_msgs) - 1:
            curr_diff = abs(depth_msgs[depth_idx][0] - color_ts)
            next_diff = abs(depth_msgs[depth_idx + 1][0] - color_ts)
            
            if next_diff < curr_diff:
                depth_idx += 1
            else:
                break
        
        # Check if the match is within the time threshold
        if depth_idx < len(depth_msgs):
            time_diff = abs(depth_msgs[depth_idx][0] - color_ts)
            if time_diff <= max_time_diff_ns:
                pairs.append((color_idx, depth_idx))
    
    return pairs


def create_gaussian_distribution(timestamps: np.ndarray, 
                                 target_timestamps: List[int],
                                 sigma_ns: int) -> np.ndarray:
    """
    Create a probability distribution by summing Gaussians centered at target timestamps.
    
    Args:
        timestamps: Array of all available timestamps (in nanoseconds)
        target_timestamps: List of timestamps to center Gaussians on (in nanoseconds)
        sigma_ns: Standard deviation of Gaussians in nanoseconds
        
    Returns:
        Normalized probability distribution (same length as timestamps)
    """
    # Initialize probability distribution
    prob_dist = np.zeros(len(timestamps))
    
    # Add a Gaussian for each target timestamp
    for target_ts in target_timestamps:
        # Gaussian: exp(-0.5 * ((x - mu) / sigma)^2)
        gaussian = np.exp(-0.5 * ((timestamps - target_ts) / sigma_ns) ** 2)
        prob_dist += gaussian
    
    # Normalize so total probability is 1
    prob_dist /= prob_dist.sum()
    
    return prob_dist


def sample_with_distribution(n_samples: int, probabilities: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Sample indices according to a probability distribution.
    
    Args:
        n_samples: Number of samples to draw
        probabilities: Probability distribution (must sum to 1)
        seed: Random seed for reproducibility
        
    Returns:
        Array of sampled indices
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.arange(len(probabilities))
    sampled_indices = np.random.choice(indices, size=n_samples, replace=False, p=probabilities)
    
    return sampled_indices


def enforce_minimum_time_gap(sampled_indices: np.ndarray, 
                            timestamps: np.ndarray,
                            min_gap_ns: int) -> np.ndarray:
    """
    Filter sampled indices to enforce a minimum time gap between consecutive samples.
    
    This implements "hard sampling" to avoid temporal redundancy.
    
    Args:
        sampled_indices: Array of sampled indices from Gaussian distribution
        timestamps: Array of timestamps corresponding to the indices
        min_gap_ns: Minimum time gap in nanoseconds
        
    Returns:
        Filtered array of indices with minimum time gap enforced
    """
    if len(sampled_indices) == 0:
        return sampled_indices
    
    # Get timestamps for sampled indices and sort by time
    sample_times = [(idx, timestamps[idx]) for idx in sampled_indices]
    sample_times.sort(key=lambda x: x[1])  # Sort by timestamp
    
    # Keep first sample, then only keep samples that are min_gap_ns apart
    filtered = [sample_times[0][0]]  # Keep first sample (index)
    last_kept_time = sample_times[0][1]
    
    for idx, ts in sample_times[1:]:
        if ts - last_kept_time >= min_gap_ns:
            filtered.append(idx)
            last_kept_time = ts
    
    return np.array(filtered)


def extract_frames(mcap_path: str, 
                   color_topic: str, 
                   depth_topic: str,
                   num_frames: int,
                   sampling_mode: str = 'random',
                   target_timestamps: Optional[List[int]] = None,
                   sigma_seconds: float = 5.0,
                   min_gap_seconds: Optional[float] = None,
                   output_base_dir: Optional[str] = None,
                   max_time_diff_ms: float = 50.0,
                   seed: Optional[int] = None):
    """
    Extract color and depth image pairs from an MCAP file.
    
    Args:
        mcap_path: Path to the MCAP file
        color_topic: Topic name for color images
        depth_topic: Topic name for depth images
        num_frames: Number of frames to extract
        sampling_mode: Sampling strategy ('random', 'gaussian', 'hard')
        target_timestamps: List of timestamps (in nanoseconds) for gaussian/hard modes
        sigma_seconds: Standard deviation of Gaussians in seconds (for gaussian/hard modes)
        min_gap_seconds: Minimum time gap between frames in seconds (for hard mode)
        output_base_dir: Base directory for output (default: data/processed_imgs)
        max_time_diff_ms: Maximum time difference between color and depth in milliseconds
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Validate sampling mode
    if sampling_mode not in ['random', 'gaussian', 'hard']:
        raise ValueError(f"Invalid sampling mode: {sampling_mode}. Must be 'random', 'gaussian', or 'hard'")
    
    # Validate required parameters for non-random modes
    if sampling_mode in ['gaussian', 'hard'] and not target_timestamps:
        raise ValueError(f"sampling_mode '{sampling_mode}' requires target_timestamps")
    
    mcap_path = Path(mcap_path)
    
    # Set default output base directory
    if output_base_dir is None:
        # Find project root and use data/processed_imgs
        project_root = find_project_root()
        output_base_dir = project_root / "data" / "processed_imgs"
    else:
        output_base_dir = Path(output_base_dir)
    
    # Create a unique timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mcap_name = mcap_path.stem
    output_dir = output_base_dir / f"{mcap_name}_{sampling_mode}_{timestamp}"
    
    # Create output directories
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading MCAP file: {mcap_path}")
    print(f"Color topic: {color_topic}")
    print(f"Depth topic: {depth_topic}")
    print(f"Sampling mode: {sampling_mode}")
    if sampling_mode in ['gaussian', 'hard']:
        print(f"Gaussian centers: {len(target_timestamps)} timestamps")
        print(f"Gaussian sigma: {sigma_seconds} seconds")
    if sampling_mode == 'hard':
        print(f"Minimum time gap: {min_gap_seconds} seconds")
    
    # Read all messages from both topics
    color_msgs = []
    depth_msgs = []
    
    # Create decoder factory for ROS2 messages
    decoder_factory = DecoderFactory()
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[decoder_factory])
        
        # Get summary info
        summary = reader.get_summary()
        if summary and summary.schemas:
            print(f"\nAvailable topics in bag:")
            topics_seen = set()
            for channel in summary.channels.values():
                if channel.topic not in topics_seen:
                    topics_seen.add(channel.topic)
                    schema = summary.schemas.get(channel.schema_id)
                    schema_name = schema.name if schema else "unknown"
                    print(f"  {channel.topic}: {schema_name}")
        
        print(f"\nReading messages...")
        
        # Read all messages
        for schema, channel, message, decoded_msg in tqdm(reader.iter_decoded_messages()):
            if channel.topic == color_topic:
                color_msgs.append((message.log_time, decoded_msg))
            elif channel.topic == depth_topic:
                depth_msgs.append((message.log_time, decoded_msg))
    
    print(f"\nFound {len(color_msgs)} color images and {len(depth_msgs)} depth images")
    
    if len(color_msgs) == 0 or len(depth_msgs) == 0:
        raise ValueError("No messages found on one or both topics")
    
    # Sort by timestamp
    color_msgs.sort(key=lambda x: x[0])
    depth_msgs.sort(key=lambda x: x[0])
    
    # Find matching pairs
    print("Finding matching color-depth pairs...")
    max_time_diff_ns = int(max_time_diff_ms * 1_000_000)
    pairs = find_matching_pairs(color_msgs, depth_msgs, max_time_diff_ns)
    
    print(f"Found {len(pairs)} matching pairs")
    
    if len(pairs) == 0:
        raise ValueError("No matching pairs found. Try increasing max_time_diff_ms")
    
    # Get timestamps of all pairs
    pair_timestamps = np.array([color_msgs[pair[0]][0] for pair in pairs])
    
    # Print timestamp range
    min_ts = pair_timestamps.min()
    max_ts = pair_timestamps.max()
    duration_s = (max_ts - min_ts) / 1e9
    print(f"\nTimestamp range:")
    print(f"  Start: {min_ts}")
    print(f"  End:   {max_ts}")
    print(f"  Duration: {duration_s:.2f} seconds")
    
    # Sample pairs based on mode
    num_frames = min(num_frames, len(pairs))
    
    if sampling_mode == 'random':
        # Random sampling
        sampled_pairs = random.sample(pairs, num_frames)
        print(f"\nExtracting {num_frames} random frame pairs...")
        
    elif sampling_mode == 'gaussian':
        # Gaussian-weighted sampling
        print("\nCreating Gaussian probability distribution...")
        sigma_ns = int(sigma_seconds * 1e9)
        prob_dist = create_gaussian_distribution(pair_timestamps, target_timestamps, sigma_ns)
        
        sampled_indices = sample_with_distribution(num_frames, prob_dist, seed)
        sampled_pairs = [pairs[i] for i in sampled_indices]
        print(f"\nExtracting {num_frames} frames using Gaussian sampling...")
        
    else:  # hard sampling
        # Gaussian sampling with minimum time gap enforcement
        print("\nCreating Gaussian probability distribution...")
        sigma_ns = int(sigma_seconds * 1e9)
        prob_dist = create_gaussian_distribution(pair_timestamps, target_timestamps, sigma_ns)
        
        # Oversample to account for filtering
        oversample_factor = 3
        initial_samples = min(num_frames * oversample_factor, len(pairs))
        
        print(f"Initial sampling (oversampling by {oversample_factor}x): {initial_samples} frames")
        sampled_indices = sample_with_distribution(initial_samples, prob_dist, seed)
        
        # Enforce minimum time gap
        min_gap_ns = int(min_gap_seconds * 1e9)
        print(f"Enforcing minimum time gap of {min_gap_seconds}s between frames...")
        filtered_indices = enforce_minimum_time_gap(sampled_indices, pair_timestamps, min_gap_ns)
        
        # Trim to desired number if we got more than requested
        if len(filtered_indices) > num_frames:
            filtered_indices = filtered_indices[:num_frames]
        
        sampled_pairs = [pairs[i] for i in filtered_indices]
        
        print(f"\n✓ After hard sampling: {len(sampled_pairs)} frames (removed {initial_samples - len(sampled_pairs)} temporally redundant frames)")
        
        if len(sampled_pairs) < num_frames * 0.8:
            print(f"\n⚠ Warning: Got {len(sampled_pairs)} frames, which is less than 80% of requested {num_frames}")
            print(f"   Consider: (1) reducing min_gap_seconds, (2) increasing sigma_seconds, or (3) requesting fewer frames")
        
        print(f"\nExtracting {len(sampled_pairs)} frames...")
    
    # Extract and save images
    for idx, (color_idx, depth_idx) in enumerate(tqdm(sampled_pairs)):
        color_ts, color_msg = color_msgs[color_idx]
        depth_ts, depth_msg = depth_msgs[depth_idx]
        
        # Decode images
        try:
            color_img = decode_image(color_msg)
            depth_img = decode_image(depth_msg)
            
            # Save images with timestamp-based filenames
            color_filename = color_dir / f"frame_{idx:06d}_{color_ts}.png"
            depth_filename = depth_dir / f"frame_{idx:06d}_{depth_ts}.png"
            
            cv2.imwrite(str(color_filename), color_img)
            
            # For depth images, save as 16-bit if possible
            if depth_img.dtype == np.uint16:
                cv2.imwrite(str(depth_filename), depth_img)
            elif depth_img.dtype == np.float32:
                # Convert float depth to uint16 (assuming meters, scale to mm)
                depth_mm = (depth_img * 1000).astype(np.uint16)
                cv2.imwrite(str(depth_filename), depth_mm)
            else:
                cv2.imwrite(str(depth_filename), depth_img)
            
        except Exception as e:
            print(f"\nError processing frame {idx}: {e}")
            continue
    
    print(f"\n✓ Done! Images saved to:")
    print(f"  Output directory: {output_dir}")
    print(f"  Color: {color_dir}")
    print(f"  Depth: {depth_dir}")
    if sampling_mode == 'hard':
        print(f"\n💡 Next step: Use data augmentation (flip, crop, rotate, color jitter) during training")
        print(f"   to build robustness without temporal redundancy!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract color and depth image pairs from MCAP ROS bags with flexible sampling strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Modes:
  random   : Random sampling across entire dataset (default)
  gaussian : Gaussian-weighted sampling around target timestamps
  hard     : Gaussian sampling with minimum time gap enforcement

Examples:
  # Random sampling (default)
  %(prog)s bag.mcap -n 100
  
  # Gaussian sampling around specific timestamps
  %(prog)s bag.mcap -n 100 --mode gaussian -t 1760644700000000000 1760644750000000000
  
  # Hard sampling with minimum time gap
  %(prog)s bag.mcap -n 100 --mode hard -t 1760644700000000000 --min-gap 0.5
  
  # Using a config file
  %(prog)s bag.mcap --config config/my_config.json
  
  # Override config settings
  %(prog)s bag.mcap --config config/my_config.json -n 200 --sigma 10.0
        """
    )
    parser.add_argument(
        "mcap_path",
        type=str,
        help="Path to the MCAP file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. Command line args override config values."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['random', 'gaussian', 'hard'],
        default=None,
        help="Sampling mode: random (default), gaussian, or hard"
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=None,
        help="Number of frames to extract (default: 100, or from config)"
    )
    parser.add_argument(
        "-t", "--timestamps",
        type=int,
        nargs='+',
        default=None,
        help="Target timestamps in nanoseconds (required for gaussian/hard modes unless in config)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Standard deviation of Gaussians in seconds (default: 5.0, or from config)"
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=None,
        help="Minimum time gap between frames in seconds for hard mode (default: 0.5, or from config)"
    )
    parser.add_argument(
        "-o", "--output-base-dir",
        type=str,
        default=None,
        help="Base output directory (default: data/processed_imgs)"
    )
    parser.add_argument(
        "-c", "--color-topic",
        type=str,
        default=None,
        help="Color image topic (default: /BD03/d455/color/image_raw, or from config)"
    )
    parser.add_argument(
        "-d", "--depth-topic",
        type=str,
        default=None,
        help="Depth image topic (default: /BD03/d455/depth/image_rect_raw, or from config)"
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=None,
        help="Maximum time difference between color and depth in milliseconds (default: 50.0, or from config)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
        if 'description' in config:
            print(f"Description: {config['description']}")
    
    # Merge config with command line args (command line takes precedence)
    sampling_mode = args.mode if args.mode else config.get('sampling_mode', 'random')
    num_frames = args.num_frames if args.num_frames is not None else config.get('num_frames', 100)
    timestamps = args.timestamps if args.timestamps else config.get('timestamps')
    sigma_seconds = args.sigma if args.sigma is not None else config.get('sigma_seconds', 5.0)
    min_gap_seconds = args.min_gap if args.min_gap is not None else config.get('min_gap_seconds', 0.5)
    color_topic = args.color_topic if args.color_topic else config.get('color_topic', '/BD03/d455/color/image_raw')
    depth_topic = args.depth_topic if args.depth_topic else config.get('depth_topic', '/BD03/d455/depth/image_rect_raw')
    max_time_diff = args.max_time_diff if args.max_time_diff is not None else config.get('max_time_diff_ms', 50.0)
    
    # Validate that we have timestamps for gaussian/hard modes
    if sampling_mode in ['gaussian', 'hard'] and not timestamps:
        parser.error(f"sampling_mode '{sampling_mode}' requires --timestamps or --config with timestamps")
    
    extract_frames(
        mcap_path=args.mcap_path,
        color_topic=color_topic,
        depth_topic=depth_topic,
        num_frames=num_frames,
        sampling_mode=sampling_mode,
        target_timestamps=timestamps,
        sigma_seconds=sigma_seconds,
        min_gap_seconds=min_gap_seconds,
        output_base_dir=args.output_base_dir,
        max_time_diff_ms=max_time_diff,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
