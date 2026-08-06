import argparse
import csv
import os
from typing import Iterable, Optional

import h5py


PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPEG_SIG = b"\xff\xd8\xff"


def detect_image_ext(b: bytes) -> str:
    if len(b) >= 8 and b[:8] == PNG_SIG:
        return "png"
    if len(b) >= 3 and b[:3] == JPEG_SIG:
        return "jpg"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    # Fallback to jpg as most deployments use JPEG
    return "jpg"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all frames from an HDF5 log to image files",
        epilog=(
            "Example:\n"
            "  python examples/utils/export_images.py logs/run.h5 --groups front_frames,rear_frames --outdir exported_frames\n"
        ),
    )
    parser.add_argument("log", help="Path to HDF5 log")
    parser.add_argument(
        "--groups",
        default="front_frames,rear_frames",
        help="Comma-separated list of frame groups to export (default: front_frames,rear_frames)",
    )
    parser.add_argument("--outdir", default="exported_frames", help="Output directory for exported images")
    parser.add_argument(
        "--naming",
        choices=["index", "timestamp"],
        default="index",
        help="Filename uses index or timestamp (default: index)",
    )
    parser.add_argument(
        "--index-csv",
        action="store_true",
        help="Write an index.csv mapping filenames to timestamps in each group directory",
    )
    return parser.parse_args(argv)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_group(grp: h5py.Group, outdir: str, naming: str, write_index: bool) -> int:
    data = grp.get("data")
    ts_ds = grp.get("timestamps")
    if data is None or ts_ds is None or data.shape[0] == 0:
        return 0

    entries = int(data.shape[0])
    rows: list[tuple[str, float]] = []
    for i in range(entries):
        b = bytes(data[i])
        ts = float(ts_ds[i])
        ext = detect_image_ext(b)
        if naming == "timestamp":
            fname = f"{ts:.3f}.{ext}"
        else:
            fname = f"{i:05d}.{ext}"
        fpath = os.path.join(outdir, fname)
        with open(fpath, "wb") as fp:
            fp.write(b)
        if write_index:
            rows.append((fname, ts))

    if write_index and rows:
        with open(os.path.join(outdir, "index.csv"), "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "timestamp"])
            writer.writerows(rows)
    return entries


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    groups: Iterable[str] = [g.strip() for g in args.groups.split(",") if g.strip()]
    count_total = 0

    with h5py.File(args.log, "r") as f:
        for gname in groups:
            if gname not in f:
                continue
            grp = f[gname]
            gout = os.path.join(args.outdir, gname)
            ensure_dir(gout)
            n = export_group(grp, gout, args.naming, args.index_csv)
            count_total += n
            print(f"Exported {n} frames from {gname} -> {gout}")

    if count_total == 0:
        print("No frames exported (check groups or log file)")
    else:
        print(f"Done. Total frames exported: {count_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


