#!/usr/bin/env python3
"""Parameter sweep for traversability on obstacle vs open frames."""
import sys, numpy as np, h5py, math
from io import BytesIO
from PIL import Image

sys.path.insert(0, 'src')
from depth_estimator import DepthEstimator

de = DepthEstimator(model_size='small')

frames = [
    ('test_outdoor_4.h5', 1696, 'person+dog', 'OBS'),
    ('test_outdoor_2.h5', 272,  'person_ahead', 'OBS'),
    ('test_outdoor_4.h5', 848,  'bushes_sides', 'OBS'),
    ('test_outdoor_1.h5', 1121, 'tree_right', 'OBS'),
    ('test_outdoor_1.h5', 0,    'open_1', 'OPEN'),
    ('test_outdoor_1.h5', 500,  'open_2', 'OPEN'),
    ('test_outdoor_3.h5', 500,  'open_3', 'OPEN'),
    ('test_outdoor_3.h5', 1500, 'open_4', 'OPEN'),
]

resolutions = [(120, 160), (240, 320)]
percentiles = [1, 5, 10, 25]
crop_top, crop_bot = 0.15, 0.60
num_bins = 16

print(f"{'label':<16} {'cat':<5} {'res':<8} {'pctl':<5} {'fwd':>6} {'minb':>6} {'<3m':>4} {'<5m':>4} {'<8m':>4}")
print("-" * 70)

for fname, idx, label, cat in frames:
    f = h5py.File(f'test_outdoor/{fname}', 'r')
    raw = f['front_frames/data'][idx]
    frame = np.array(Image.open(BytesIO(bytes(raw))).convert('RGB'), dtype=np.uint8)
    f.close()

    for res in resolutions:
        depth = de.estimate(frame, target_size=res)
        h, w = depth.shape
        r0, r1 = int(h * crop_top), int(h * crop_bot)
        band = depth[r0:r1, :]

        fov_rad = math.radians(90.0)
        fx = w / (2.0 * math.tan(fov_rad / 2.0))
        cx = w / 2.0
        u = np.arange(w, dtype=np.float32)
        yaw = np.arctan((u - cx) / fx)
        edges = np.linspace(-fov_rad / 2, fov_rad / 2, num_bins + 1)

        for pctl in percentiles:
            clearance = np.full(num_bins, 80.0)
            for b in range(num_bins):
                mask = (yaw >= edges[b]) & (yaw < edges[b + 1])
                cols = np.where(mask)[0]
                if len(cols) > 0:
                    vals = band[:, cols]
                    valid = vals[vals > 0]
                    if len(valid) > 0:
                        clearance[b] = float(np.percentile(valid, pctl))

            center = num_bins // 2
            fwd = float(np.min(clearance[center - 1 : center + 2]))
            minbin = float(np.min(clearance))
            blk3 = int(np.sum(clearance < 3.0))
            blk5 = int(np.sum(clearance < 5.0))
            blk8 = int(np.sum(clearance < 8.0))

            print(
                f"{label:<16} {cat:<5} {res[0]:>3}x{res[1]:<4} p{pctl:<4} "
                f"{fwd:6.1f} {minbin:6.1f} {blk3:4d} {blk5:4d} {blk8:4d}"
            )
    print()
