import argparse
import os
import sys
from typing import Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt  
import cv2


class ExampleArgumentParser(argparse.ArgumentParser):
    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{os.path.basename(__file__)}: error: {message}")
        print("Example:")
        print("  python examples/utils/analyze_log.py logs/rover_log_20250101_120000.h5 --save-plots --outdir plots")
        self.exit(2)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = ExampleArgumentParser(
        description="Analyze Earth Rovers HDF5 log",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python examples/utils/analyze_log.py logs/rover_log_20250101_120000.h5 --save-plots --outdir plots\n"
        ),
    )
    parser.add_argument("path", help="Path to HDF5 log (e.g., logs/rover_log_*.h5)")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots")
    parser.add_argument("--outdir", default="plots", help="Directory to save plots (default: plots)")
    return parser.parse_args(argv)


def describe(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: empty")
        return
    if arr.dtype.fields is None:
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
        return
    fields = list(arr.dtype.fields.keys())
    print(f"{name}: n={arr.shape[0]} fields={fields}")
    # Print simple stats for common fields
    for fld in ("battery", "speed", "gps_signal", "vibration", "latitude", "longitude"):
        if fld in fields:
            col = arr[fld]
            finite = np.isfinite(col)
            if finite.any():
                vals = col[finite]
                print(
                    f"  {fld}: min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f} median={np.median(vals):.3f}"
                )


def plot_if_available(h5: h5py.File, outdir: str, enable: bool) -> None:
    if not enable:
        return
    os.makedirs(outdir, exist_ok=True)

    # Telemetry time series
    tele = h5["telemetry"][:]
    if tele.size > 0:
        t0 = float(tele["timestamp"][0])
        tt = tele["timestamp"] - t0
        for fld in ("battery", "speed", "gps_signal", "vibration"):
            plt.figure()
            plt.plot(tt, tele[fld])
            plt.xlabel("time (s)")
            plt.ylabel(fld)
            plt.title(f"telemetry.{fld}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        plt.savefig(os.path.join(outdir, "telemetry.png"))
        plt.close("all")

    # Path (lat/lon)
    if tele.size > 0 and np.isfinite(tele["latitude"]).any():
        plt.figure()
        lat = tele["latitude"]
        lon = tele["longitude"]
        m = np.isfinite(lat) & np.isfinite(lon)
        plt.plot(lon[m], lat[m], ".-", alpha=0.7)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.title("GPS path")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "path.png"))
        plt.close()

    # IMU magnitudes
    def plot_imu(name: str) -> None:
        ds = h5[name][:]
        if ds.size == 0:
            return
        t0 = float(ds["t"][0])
        tt = ds["t"] - t0
        plt.figure()
        plt.plot(tt, ds["x"], label="x", alpha=0.8)
        plt.plot(tt, ds["y"], label="y", alpha=0.8)
        plt.plot(tt, ds["z"], label="z", alpha=0.8)
        plt.xlabel("time (s)")
        plt.ylabel(name)
        plt.title(f"{name} components")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{name}.png"))
        plt.close()

    for nm in ("accels", "gyros", "mags"):
        plot_imu(nm)

    # RPMs
    rpms = h5["rpms"][:]
    if rpms.size > 0:
        t0 = float(rpms["t"][0])
        tt = rpms["t"] - t0
        plt.figure()
        for fld in ("front_left", "front_right", "rear_left", "rear_right"):
            plt.plot(tt, rpms[fld], label=fld, alpha=0.8)
        plt.xlabel("time (s)")
        plt.ylabel("rpm")
        plt.title("Wheel RPMs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "rpms.png"))
        plt.close()

    # Controls
    if "controls" in h5:
        ctrl = h5["controls"][:]
        if ctrl.size > 0:
            t0 = float(ctrl["timestamp"][0])
            tt = ctrl["timestamp"] - t0
            plt.figure()
            plt.plot(tt, ctrl["linear"], label="linear", alpha=0.9)
            plt.plot(tt, ctrl["angular"], label="angular", alpha=0.9)
            plt.xlabel("time (s)")
            plt.ylabel("command")
            plt.title("Commanded velocities")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "controls.png"))
            plt.close()

    # Save a sample front/rear frame if present
    def save_sample(group_name: str, fname: str) -> None:
        if group_name not in h5:
            return
        grp = h5[group_name]
        if grp["data"].shape[0] == 0:
            return
        # Take the last frame
        b = bytes(grp["data"][-1])
        img = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            return
        cv2.imwrite(os.path.join(outdir, fname), img)

    save_sample("front_frames", "front_sample.jpg")
    save_sample("rear_frames", "rear_sample.jpg")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        print("Example:")
        print("  python examples/analyze_log.py logs/rover_log_20250101_120000.h5 --save-plots --outdir plots")
        raise SystemExit(2)

    with h5py.File(args.path, "r") as f:
        for name in ("telemetry", "accels", "gyros", "mags", "rpms"):
            if name in f:
                arr = f[name][:]
                describe(name, arr)
            else:
                print(f"{name}: not found")

        plot_if_available(f, args.outdir, args.save_plots)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


