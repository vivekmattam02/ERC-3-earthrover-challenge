#!/usr/bin/env python3
"""Evaluate temporal localization on a sequence of query frames.

This script runs retrieval frame-by-frame and then stabilizes the result with
TemporalLocalizer. It is meant as the first practical check for repeated
corridor localization before wiring the logic into live runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from baseline import (
    descriptor_distance_search,
    get_device,
    load_cosplace_model,
    load_descriptor_archive,
    load_descriptor_config,
    make_cosplace_transform,
    preprocess_image,
)
from temporal_localization import TemporalLocalizer, TemporalLocalizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal localization on a corridor image sequence.")
    parser.add_argument("--database", type=Path, required=True, help="Path to descriptors.npz from baseline build-db.")
    parser.add_argument("--query-dir", type=Path, required=True, help="Directory of query images.")
    parser.add_argument("--query-data-info-json", type=Path, default=None, help="Optional query data_info.json with headings.")
    parser.add_argument("--start-step", type=int, default=0, help="First query step to evaluate.")
    parser.add_argument("--end-step", type=int, default=None, help="Last query step to evaluate (exclusive).")
    parser.add_argument("--stride", type=int, default=1, help="Query every Nth frame.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval candidates to score per frame.")
    parser.add_argument("--max-step-jump", type=int, default=15, help="Maximum unpenalized node jump per update.")
    parser.add_argument("--jump-penalty", type=float, default=0.05, help="Penalty per node beyond allowed jump.")
    parser.add_argument("--backward-penalty", type=float, default=0.15, help="Penalty for backward motion in node index.")
    parser.add_argument("--heading-penalty", type=float, default=0.002, help="Penalty scale per degree of heading mismatch.")
    parser.add_argument("--ambiguity-margin", type=float, default=0.05, help="Hold previous node if best and second scores are too close.")
    parser.add_argument("--results-json", type=Path, default=None, help="Optional JSON output for detailed per-frame results.")
    return parser.parse_args()


def load_data_info(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_heading_lookup(data_info: list[dict]) -> tuple[dict[int, float], dict[str, float], dict[str, int]]:
    node_heading_by_step: dict[int, float] = {}
    heading_by_image: dict[str, float] = {}
    step_by_image: dict[str, int] = {}
    for entry in data_info:
        step = int(entry["step"])
        image = entry["image"]
        orientation = entry.get("orientation")
        if orientation is not None:
            heading = float(orientation)
            node_heading_by_step[step] = heading
            heading_by_image[image] = heading
        step_by_image[image] = step
    return node_heading_by_step, heading_by_image, step_by_image


def build_metadata_by_image(data_info: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for entry in data_info:
        metadata[entry["image"]] = entry
    return metadata


def is_stationary(metadata: dict | None) -> bool:
    if metadata is None:
        return False
    speed = metadata.get("speed")
    linear = metadata.get("linear")
    angular = metadata.get("angular")

    speed = 0.0 if speed is None else float(speed)
    linear = 0.0 if linear is None else float(linear)
    angular = 0.0 if angular is None else float(angular)

    return abs(speed) <= 0.05 and abs(linear) <= 0.02 and abs(angular) <= 0.02


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    device = get_device()
    descriptor_config = load_descriptor_config(args.database)
    descriptors, image_names, _ = load_descriptor_archive(args.database)
    model = load_cosplace_model(None, descriptor_config, device)
    transform = make_cosplace_transform(descriptor_config)

    data_info = load_data_info(args.query_data_info_json)
    _, heading_by_image, _ = build_heading_lookup(data_info)
    metadata_by_image = build_metadata_by_image(data_info)

    db_step_lookup = {}
    db_heading_lookup = {}
    for idx, image_name in enumerate(image_names):
        try:
            step = int(Path(image_name).stem)
            db_step_lookup[idx] = step
            if image_name in heading_by_image:
                db_heading_lookup[idx] = heading_by_image[image_name]
        except ValueError:
            continue

    image_paths = sorted(args.query_dir.glob("*.png"))
    if not image_paths:
        image_paths = sorted(args.query_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No query images found in {args.query_dir}")

    end_step = args.end_step if args.end_step is not None else len(image_paths)
    selected = image_paths[args.start_step:end_step:args.stride]
    if not selected:
        raise RuntimeError("No query images selected.")

    localizer = TemporalLocalizer(
        TemporalLocalizerConfig(
            top_k=args.top_k,
            max_step_jump=args.max_step_jump,
            jump_penalty=args.jump_penalty,
            backward_penalty=args.backward_penalty,
            heading_penalty=args.heading_penalty,
            ambiguity_margin=args.ambiguity_margin,
        )
    )

    records: list[dict] = []
    exact = 0
    near = 0
    moving_exact = 0
    moving_near = 0
    moving_total = 0
    stationary_total = 0

    for query_path in selected:
        query_name = query_path.name
        query_step = int(query_path.stem)
        query_heading = heading_by_image.get(query_name)
        query_metadata = metadata_by_image.get(query_name)
        stationary = is_stationary(query_metadata)

        query_tensor = preprocess_image(query_path, transform, descriptor_config).to(device)
        with torch.no_grad():
            query_desc = model(query_tensor)
            query_desc = torch.nn.functional.normalize(query_desc, p=2, dim=1)
        query_desc_np = query_desc.cpu().numpy()[0].astype("float32")

        candidates = descriptor_distance_search(descriptors, query_desc_np, top_k=args.top_k)
        candidate_rows = [
            {
                "index": int(index),
                "distance": float(distance),
                "image_name": image_names[index],
                "step": db_step_lookup.get(int(index)),
            }
            for index, distance in candidates
        ]

        state = localizer.update(
            candidate_rows,
            observation_heading=query_heading,
            node_heading_lookup=db_heading_lookup,
        )

        predicted_step = db_step_lookup.get(int(state["node_index"])) if state["node_index"] is not None else None
        step_error = abs(predicted_step - query_step) if predicted_step is not None else None
        if step_error == 0:
            exact += 1
        if step_error is not None and step_error <= max(2, args.stride * 2):
            near += 1
        if stationary:
            stationary_total += 1
        else:
            moving_total += 1
            if step_error == 0:
                moving_exact += 1
            if step_error is not None and step_error <= max(2, args.stride * 2):
                moving_near += 1

        records.append(
            {
                "query_image": query_name,
                "query_step": query_step,
                "query_heading": query_heading,
                "query_speed": None if query_metadata is None else query_metadata.get("speed"),
                "query_linear": None if query_metadata is None else query_metadata.get("linear"),
                "query_angular": None if query_metadata is None else query_metadata.get("angular"),
                "stationary": stationary,
                "predicted_node_index": state["node_index"],
                "predicted_step": predicted_step,
                "step_error": step_error,
                "confidence": state["confidence"],
                "stable_steps": state.get("stable_steps"),
                "held_previous": state.get("held_previous"),
                "reason": state.get("reason"),
                "best_candidate": state.get("best_candidate"),
                "second_candidate": state.get("second_candidate"),
            }
        )

    total = len(records)
    summary = {
        "num_queries": total,
        "exact_match_count": exact,
        "exact_match_rate": exact / total if total else 0.0,
        "near_match_count": near,
        "near_match_rate": near / total if total else 0.0,
        "moving_query_count": moving_total,
        "moving_exact_match_count": moving_exact,
        "moving_exact_match_rate": moving_exact / moving_total if moving_total else 0.0,
        "moving_near_match_count": moving_near,
        "moving_near_match_rate": moving_near / moving_total if moving_total else 0.0,
        "stationary_query_count": stationary_total,
        "start_step": args.start_step,
        "end_step": end_step,
        "stride": args.stride,
    }

    print(json.dumps(summary, indent=2))
    print("\nSample results:")
    for record in records[:5]:
        print(
            f"{record['query_image']} -> step {record['predicted_step']} "
            f"(err={record['step_error']}, conf={record['confidence']:.3f}, held={record['held_previous']})"
        )

    if args.results_json is not None:
        payload = {
            "summary": summary,
            "records": records,
        }
        with args.results_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved temporal localization results to {args.results_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
