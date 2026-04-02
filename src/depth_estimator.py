"""
Depth Estimator

Wrapper around Depth Anything V2 for monocular depth estimation.
Used for optional runtime depth-aware safety checking.

Usage:
    from depth_estimator import DepthEstimator

    estimator = DepthEstimator()
    depth_map = estimator.estimate(rgb_frame)  # Returns (H, W) in meters

Author: Vivek Mattam
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add Depth-Anything-V2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'Depth-Anything-V2', 'metric_depth'))


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.

    Estimates metric depth (in meters) from a single RGB image.
    """

    def __init__(self, model_size='small', device=None, max_depth=None, checkpoint_domain=None):
        """
        Initialize depth estimator.

        Args:
            model_size: 'small', 'base', or 'large'
                - small: fastest, ~25M params (recommended for robot)
                - base: balanced, ~97M params
                - large: most accurate, ~335M params
            device: 'cuda' or 'cpu'. Auto-detects if None.
            max_depth: Maximum depth in meters.  If None, auto-detected from
                the checkpoint name (vkitti=80, hypersim=20, fallback=20).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_size = model_size

        # Find checkpoint first so we can infer max_depth from it.
        checkpoint_path = self._find_checkpoint(model_size, checkpoint_domain=checkpoint_domain)
        if max_depth is None:
            max_depth = self._infer_max_depth(checkpoint_path)
        self.max_depth = max_depth

        print(f"Loading Depth Anything V2 ({model_size}) on {self.device}, max_depth={self.max_depth}...")

        # Model configurations
        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if model_size not in model_configs:
            raise ValueError(f"model_size must be 'small', 'base', or 'large', got '{model_size}'")

        config = model_configs[model_size]

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            # Try alternative import path
            da_path = os.path.join(os.path.dirname(__file__), '..', 'third_party',
                                   'Depth-Anything-V2')
            sys.path.insert(0, da_path)
            from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth
        )

        # Load checkpoint
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"WARNING: No checkpoint found for {model_size} model.")
            print(f"Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small")
            print(f"Place in: third_party/Depth-Anything-V2/checkpoints/")

        self.model.eval()
        self.model.to(self.device)
        print("Depth estimator ready!")

    def _find_checkpoint(self, model_size, checkpoint_domain=None):
        """Search for checkpoint file in common locations."""
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        ckpt_dir = os.path.join(base_dir, 'third_party', 'Depth-Anything-V2', 'checkpoints')
        # Map model_size → ViT encoder suffix used in actual HuggingFace filenames
        encoder_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
        enc = encoder_map.get(model_size, model_size)
        metric_search = [
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_vkitti_{enc}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_hypersim_{enc}.pth'),
        ]
        if checkpoint_domain == 'indoor':
            metric_search = [metric_search[1], metric_search[0]]
        elif checkpoint_domain == 'outdoor':
            metric_search = [metric_search[0], metric_search[1]]

        search_paths = metric_search + [
            # Legacy / renamed variants
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_{model_size}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_{model_size}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_{enc}.pth'),
            os.path.join(base_dir, 'models', f'depth_anything_v2_{model_size}.pth'),
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _infer_max_depth(checkpoint_path) -> float:
        """Infer the correct max_depth from checkpoint filename.

        Per the Depth Anything V2 metric_depth README:
          - vkitti  (outdoor driving) → max_depth = 80
          - hypersim (indoor scenes)  → max_depth = 20
        """
        if checkpoint_path is not None:
            name = os.path.basename(checkpoint_path).lower()
            if "vkitti" in name:
                return 80.0
            if "hypersim" in name:
                return 20.0
        return 20.0  # conservative fallback

    def estimate(self, image, target_size=None):
        """
        Estimate depth from RGB image.

        Args:
            image: RGB image as numpy array (H, W, 3) with values 0-255
            target_size: Optional (H, W) to resize output. If None, matches input.

        Returns:
            depth_map: numpy array (H, W) with depth in meters
        """
        h, w = image.shape[:2]

        if target_size is None:
            target_size = (h, w)

        # Preprocess
        img_tensor = self._preprocess(image)

        # Inference
        with torch.no_grad():
            depth = self.model(img_tensor)

        # Post-process
        depth = depth.squeeze().cpu().numpy()

        # Resize to target size if needed
        if depth.shape != target_size:
            depth = np.array(Image.fromarray(depth).resize(
                (target_size[1], target_size[0]), Image.BILINEAR))

        return depth

    def estimate_batch(self, images):
        """
        Estimate depth for a batch of images.

        Args:
            images: numpy array (N, H, W, 3) with values 0-255

        Returns:
            depth_maps: numpy array (N, H, W) with depth in meters
        """
        target_h, target_w = images.shape[1], images.shape[2]

        batch_tensors = []
        for img in images:
            batch_tensors.append(self._preprocess(img).squeeze(0))

        batch = torch.stack(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            depths = self.model(batch)

        depths = depths.cpu().numpy()

        # Resize to match input dimensions if needed
        if depths.shape[1] != target_h or depths.shape[2] != target_w:
            resized = np.zeros((depths.shape[0], target_h, target_w), dtype=depths.dtype)
            for i in range(depths.shape[0]):
                resized[i] = np.array(Image.fromarray(depths[i]).resize(
                    (target_w, target_h), Image.BILINEAR))
            depths = resized

        return depths

    def _preprocess(self, image):
        """Preprocess image for model input."""
        # Convert to float and normalize
        img = image.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # To tensor (H, W, 3) -> (1, 3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)

        # Resize to model input size (multiples of 14 for ViT)
        h, w = img_tensor.shape[2:]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        if new_h != h or new_w != w:
            img_tensor = F.interpolate(img_tensor, size=(new_h, new_w),
                                       mode='bilinear', align_corners=False)

        return img_tensor

    def get_polar_clearance(self, depth_map, num_bins=32, crop_bottom=0.6,
                            fov_horizontal=90.0, fx=None, cx=None):
        """
        Convert depth map to polar clearance vector.

        For each angular direction,
        what is the minimum distance to an obstacle?

        Args:
            depth_map: (H, W) depth in meters
            num_bins: Number of angular bins (default 32)
            crop_bottom: Fraction of image to keep (bottom portion)
            fov_horizontal: Horizontal field of view in degrees
            fx: Focal length in pixels (auto-computed if None)
            cx: Principal point x (auto-computed if None)

        Returns:
            clearance: numpy array (num_bins,) - min depth per direction
            bin_centers: numpy array (num_bins,) - center angle of each bin
        """
        h, w = depth_map.shape

        # Crop to bottom portion (ground-level obstacles)
        crop_start = int(h * (1.0 - crop_bottom))
        depth_cropped = depth_map[crop_start:, :]

        h_crop, w_crop = depth_cropped.shape

        # Camera intrinsics
        if fx is None:
            fx = w / (2.0 * np.tan(np.radians(fov_horizontal / 2.0)))
        if cx is None:
            cx = w / 2.0

        # Compute yaw angle for each pixel column
        u = np.arange(w_crop)
        yaw_per_col = np.arctan((u - cx) / fx)  # radians

        # Define bin edges
        fov_rad = np.radians(fov_horizontal)
        bin_edges = np.linspace(-fov_rad / 2, fov_rad / 2, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Compute clearance per bin (hard min for simplicity at inference)
        clearance = np.full(num_bins, np.inf)

        for b in range(num_bins):
            # Find columns that fall in this bin
            mask = (yaw_per_col >= bin_edges[b]) & (yaw_per_col < bin_edges[b + 1])
            cols_in_bin = np.where(mask)[0]

            if len(cols_in_bin) > 0:
                # Minimum depth across all pixels in these columns
                bin_depths = depth_cropped[:, cols_in_bin]
                valid_depths = bin_depths[bin_depths > 0]  # Ignore zero/invalid
                if len(valid_depths) > 0:
                    clearance[b] = np.min(valid_depths)

        # Replace inf with max_depth
        clearance[clearance == np.inf] = self.max_depth

        return clearance, bin_centers

    def is_waypoint_safe(self, waypoint, clearance, bin_centers, margin=0.5):
        """
        Check if a waypoint direction is safe.

        Args:
            waypoint: (x, y) predicted waypoint in robot frame
            clearance: polar clearance vector from get_polar_clearance()
            bin_centers: bin center angles
            margin: safety margin in meters (default 0.5m)

        Returns:
            safe: True if clearance at waypoint direction >= margin
            clearance_at_wp: clearance value in that direction
        """
        # Get waypoint yaw angle
        wp_yaw = np.arctan2(waypoint[1], waypoint[0])

        # Find nearest bin
        bin_idx = np.argmin(np.abs(bin_centers - wp_yaw))
        clearance_at_wp = clearance[bin_idx]

        return clearance_at_wp >= margin, clearance_at_wp

    def get_safe_direction(self, clearance, bin_centers, margin=0.5):
        """
        Find the safest direction to travel.

        Args:
            clearance: polar clearance vector
            bin_centers: bin center angles
            margin: safety margin

        Returns:
            best_angle: safest direction in radians
            best_clearance: clearance in that direction
        """
        # Prefer center bins (going straight) if safe
        center_idx = len(bin_centers) // 2
        if clearance[center_idx] >= margin:
            return bin_centers[center_idx], clearance[center_idx]

        # Otherwise find bin with maximum clearance
        best_idx = np.argmax(clearance)
        return bin_centers[best_idx], clearance[best_idx]


# Quick test if run directly
if __name__ == "__main__":
    print("Testing Depth Estimator...")
    print("=" * 60)

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        estimator = DepthEstimator(model_size='small')

        # Test depth estimation
        print("\n[1] Testing depth estimation...")
        depth = estimator.estimate(dummy_image)
        print(f"  Input shape: {dummy_image.shape}")
        print(f"  Output shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")

        # Test polar clearance
        print("\n[2] Testing polar clearance...")
        clearance, bin_centers = estimator.get_polar_clearance(depth)
        print(f"  Clearance shape: {clearance.shape}")
        print(f"  Bin centers range: [{np.degrees(bin_centers[0]):.1f}, {np.degrees(bin_centers[-1]):.1f}] degrees")
        print(f"  Clearance range: [{clearance.min():.2f}, {clearance.max():.2f}] meters")

        # Test safety check
        print("\n[3] Testing waypoint safety...")
        test_wp = np.array([1.0, 0.0])  # Straight ahead
        safe, cl = estimator.is_waypoint_safe(test_wp, clearance, bin_centers)
        print(f"  Waypoint (1.0, 0.0): safe={safe}, clearance={cl:.2f}m")

        print("\nDepth estimator test passed!")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Depth Anything V2 checkpoint is downloaded.")
        print("The module structure is ready for when you have the checkpoint.")
