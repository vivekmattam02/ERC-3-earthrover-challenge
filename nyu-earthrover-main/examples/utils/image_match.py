import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np


@dataclass
class MatchResult:
    index: int
    group: str
    timestamp: float
    num_inliers: int
    score: float


def decode_h5_image(grp: h5py.Group, idx: int) -> Optional[np.ndarray]:
    if "data" not in grp or grp["data"].shape[0] == 0:
        return None
    b = bytes(grp["data"][idx])
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_target(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def compute_keypoints_descriptors(img: np.ndarray, method: str) -> Tuple[list, Optional[np.ndarray]]:
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
    elif method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT not available in this OpenCV build")
        detector = cv2.SIFT_create(nfeatures=2000)
    else:
        raise ValueError("method must be 'orb' or 'sift'")
    kp, des = detector.detectAndCompute(img, None)
    return kp, des


def match_descriptors(des1: np.ndarray, des2: np.ndarray, method: str) -> List[cv2.DMatch]:
    if method == "orb":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  # sift
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def ransac_inliers(kp1, kp2, matches: List[cv2.DMatch]) -> Tuple[int, Optional[np.ndarray]]:
    if len(matches) < 4:
        return 0, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, None
    return int(mask.sum()), H


def score_match(num_inliers: int, num_matches: int) -> float:
    if num_matches == 0:
        return 0.0
    return num_inliers + 0.1 * num_matches


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find best matching frame in HDF5 log for a target image",
        epilog=(
            "Example:\n"
            "  python examples/utils/image_match.py logs/run.h5 --target assets/axis.jpg --method sift --topk 5 --outdir matches\n"
        ),
    )
    parser.add_argument("log", help="Path to HDF5 log with frames")
    parser.add_argument("--target", required=True, help="Path to target image")
    parser.add_argument("--method", choices=["orb", "sift"], default="orb")
    parser.add_argument("--group", choices=["front_frames", "rear_frames"], default="front_frames")
    parser.add_argument("--topk", type=int, default=3, help="Show top-K matches")
    parser.add_argument("--outdir", default="matches", help="Directory to write visualization images")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    target = load_target(args.target)
    kp_t, des_t = compute_keypoints_descriptors(target, args.method)
    if des_t is None or len(kp_t) == 0:
        raise SystemExit("No features found in target image")

    results: List[MatchResult] = []
    with h5py.File(args.log, "r") as f:
        if args.group not in f or f[args.group]["data"].shape[0] == 0:
            raise SystemExit(f"No frames in group {args.group}")
        grp = f[args.group]
        n = grp["data"].shape[0]
        for i in range(n):
            img = decode_h5_image(grp, i)
            if img is None:
                continue
            kp, des = compute_keypoints_descriptors(img, args.method)
            if des is None or len(kp) == 0:
                continue
            good = match_descriptors(des_t, des, args.method)
            inliers, H = ransac_inliers(kp_t, kp, good)
            sc = score_match(inliers, len(good))
            ts = float(grp["timestamps"][i]) if "timestamps" in grp else 0.0
            results.append(MatchResult(index=i, group=args.group, timestamp=ts, num_inliers=inliers, score=sc))

            # Save a quick viz for top candidates incrementally (optional later selection)
            if inliers >= 10:
                vis = cv2.drawMatches(target, kp_t, img, kp, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(args.outdir, f"cand_{args.group}_{i:05d}.jpg"), vis)

    if not results:
        print("No matches found")
        return 0

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[: args.topk]

    print("Top matches:")
    for r in top:
        print(f"  idx={r.index} group={r.group} t={r.timestamp:.3f} inliers={r.num_inliers} score={r.score:.2f}")

    # Save best match visualization
    with h5py.File(args.log, "r") as f:
        grp = f[top[0].group]
        best = decode_h5_image(grp, top[0].index)
        if best is not None:
            kp_b, des_b = compute_keypoints_descriptors(best, args.method)
            if des_b is not None:
                good = match_descriptors(des_t, des_b, args.method)
                vis = cv2.drawMatches(target, kp_t, best, kp_b, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(args.outdir, f"best_{top[0].group}_{top[0].index:05d}.jpg"), vis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


