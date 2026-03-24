#!/usr/bin/env python3
"""ERC indoor baseline for corridor localization and graph planning.

This script turns the earlier notebook-style work into a reusable shared
baseline. It provides two main workflows:

1. `build-db`
   Build a reference image database, descriptor archive, and graph artifacts for
   a known corridor route.

2. `query`
   Localize a query image against a previously built database, with optional
   SuperPoint/SuperGlue geometric verification.

The baseline intentionally separates:
- place recognition and graph construction
- geometric verification
- action-graph construction

This is the perception/planning baseline for the ERC indoor stack. It is not
the MBRA controller and not the full EarthRover runtime loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import networkx as nx
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from networkx.readwrite import json_graph


DEFAULT_VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


############# Configuration & Argument Parsing
@dataclass
class DescriptorConfig:
    '''Configuration parameters for descriptor extraction and model loading.\n
     - backbone: The name of the CosPlace backbone architecture to use (e.g., "ResNet18").
     - fc_output_dim: The output dimension of the fully connected layer in the CosPlace model.
     - resize_height: The height to which input images will be resized before descriptor extraction.
     - resize_width: The width to which input images will be resized before descriptor extraction.
     - crop_top_ratio: The normalized vertical position (0.0 to 1.0) to start cropping the image from the top before descriptor extraction.
     - crop_bottom_ratio: The normalized vertical position (0.0 to 1.0) to end cropping the image from the bottom before descriptor extraction.
     - batch_size: The number of images to process in a batch during descriptor extraction.
    '''
    backbone: str = "ResNet18"
    '''Name of the CosPlace backbone architecture to use (e.g., "ResNet18").'''
    fc_output_dim: int = 512
    '''Output dimension of the fully connected layer.'''
    resize_height: int = 320
    '''Height to resize images to before descriptor extraction.'''
    resize_width: int = 320
    '''Width to resize images to before descriptor extraction.'''
    crop_top_ratio: float = 0.12
    '''Top crop ratio before descriptor extraction.'''
    crop_bottom_ratio: float = 0.88
    '''Bottom crop ratio before descriptor extraction.'''
    batch_size: int = 64
    '''Batch size for descriptor extraction.'''

@dataclass
class VerificationConfig:
    '''Configuration parameters for geometric verification of query-candidate matches.\n
    - fx: Focal length in the x-direction.
    - fy: Focal length in the y-direction.
    - cx: Principal point x-coordinate.
    - cy: Principal point y-coordinate.
    - top_k: The number of top retrieval candidates to verify.
    - inlier_threshold: The minimum number of inliers required to consider a match as verified'''

    fx: float = 92.0
    '''Focal length in the x-direction.'''
    fy: float = 92.0
    '''Focal length in the y-direction.'''
    cx: float = 160.0
    '''Principal point x-coordinate.'''
    cy: float = 120.0
    '''Principal point y-coordinate.'''

    top_k: int = 5
    '''Number of top retrieval candidates to verify.'''
    inlier_threshold: int = 30
    '''Minimum number of inliers required to consider a match as verified.'''


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the baseline script.
    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Baseline corridor localization and graph-planning utilities for ERC indoor."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_db = subparsers.add_parser(
        "build-db",
        help="Build a descriptor database and graph artifacts from a reference corridor traversal.",
    )
    build_db.add_argument("--image-dir", type=Path, required=True, help="Directory of reference images.")
    build_db.add_argument("--output-dir", type=Path, required=True, help="Directory to write artifacts into.")
    build_db.add_argument(
        "--cosplace-repo",
        type=Path,
        default=None,
        help="Optional local CosPlace repo path. If omitted, torch.hub will use the remote repo.",
    )
    build_db.add_argument("--backbone", default="ResNet18", help="CosPlace backbone name.")
    build_db.add_argument("--descriptor-dim", type=int, default=512, help="CosPlace output dimension.")
    build_db.add_argument("--resize-height", type=int, default=320, help="Descriptor input height.")
    build_db.add_argument("--resize-width", type=int, default=320, help="Descriptor input width.")
    build_db.add_argument("--crop-top", type=float, default=0.12, help="Top crop ratio before descriptor extraction.")
    build_db.add_argument("--crop-bottom", type=float, default=0.88, help="Bottom crop ratio before descriptor extraction.")
    build_db.add_argument("--batch-size", type=int, default=64, help="Descriptor extraction batch size.")
    build_db.add_argument("--step", type=int, default=1, help="Subsample step for input images.")
    build_db.add_argument("--limit", type=int, default=None, help="Optional max number of images to use.")
    build_db.add_argument("--knn", type=int, default=10, help="Number of retrieval neighbors to connect in the graph.")
    build_db.add_argument(
        "--data-info-json",
        type=Path,
        default=None,
        help="Optional data_info.json used to build the action graph.",
    )

    query = subparsers.add_parser(
        "query",
        help="Localize one query image against a previously built descriptor database.",
    )
    query.add_argument("--database", type=Path, required=True, help="Path to descriptors.npz from build-db.")
    query.add_argument("--query-image", type=Path, required=True, help="Query image to localize.")
    query.add_argument(
        "--cosplace-repo",
        type=Path,
        default=None,
        help="Optional local CosPlace repo path. If omitted, torch.hub will use the remote repo.",
    )
    query.add_argument("--top-k", type=int, default=5, help="Number of retrieval candidates to report.")
    query.add_argument(
        "--superglue-root",
        type=Path,
        default=None,
        help="Optional SuperGluePretrainedNetwork repo or models root for geometric verification.",
    )
    query.add_argument("--fx", type=float, default=92.0, help="Camera fx for geometric verification.")
    query.add_argument("--fy", type=float, default=92.0, help="Camera fy for geometric verification.")
    query.add_argument("--cx", type=float, default=160.0, help="Camera cx for geometric verification.")
    query.add_argument("--cy", type=float, default=120.0, help="Camera cy for geometric verification.")
    query.add_argument("--inlier-threshold", type=int, default=30, help="Minimum inliers to mark verification as strong.")
    query.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional output path for query results JSON.",
    )

    return parser.parse_args()



############### Image & File Handling

def load_data_info(json_path: Path) -> list[dict]:
    """Load and parse the data_info.json file.

    This file contains metadata linking image frames to control actions taken
    at each step of a recorded trajectory. The contents are sorted by the
    'step' number.

    Args:
        json_path (Path): Path to the data_info.json file.

    Returns:
        list[dict]: A list of dictionaries, where each entry corresponds to a
                    step in the trajectory.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        data_info = json.load(handle)
    return sorted(data_info, key=lambda item: item.get("step", 0))


def numeric_sort_key(path: Path) -> tuple[int, str]:
    """
    Generate a sorting key for numeric filenames.
    This key sorts purely numeric filenames (e.g., "123.jpg") before
    non-numeric ones (e.g., "frame_a.jpg"). Numeric filenames are sorted
    numerically, while non-numeric ones are sorted alphabetically.
    Args:
        path (Path): Path object to sort.
    Returns:
        tuple[int, str]: A sorting key where the first element is 0 for
                         numeric names and 1 for others, and the second
                         element is the padded number or the original name.
    """
    stem = path.stem
    try:
        return (0, f"{int(stem):020d}")
    except ValueError:
        return (1, stem)


def collect_image_paths(image_dir: Path, step: int = 1, limit: Optional[int] = None,
    valid_exts: Iterable[str] = DEFAULT_VALID_EXTS) -> list[Path]:
    """
    Collect image file paths from a directory, optionally subsampling and limiting.
    Args:
        image_dir (Path): Directory containing images.
        step (int): Subsampling step.
        limit (Optional[int]): Max number of images.
        valid_exts (Iterable[str]): Valid image extensions.
    Returns:
        list[Path]: List of image paths.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if step < 1:
        raise ValueError("step must be >= 1")

    valid_exts = tuple(ext.lower() for ext in valid_exts)
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts], key=numeric_sort_key,)
    if limit is not None:
        image_paths = image_paths[:limit]
    image_paths = image_paths[::step]
    if not image_paths:
        raise RuntimeError(f"No valid images found in {image_dir}")
    return image_paths


def load_descriptor_archive(npz_path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load the descriptor database from a .npz archive file.

    Args:
        npz_path (Path): Path to the 'descriptors.npz' file.

    Returns:
        tuple[np.ndarray, list[str], list[str]]: A tuple containing:
            - The descriptor array as a float32 NumPy array.
            - The list of image base names.
            - The list of resolved image path strings.
    """
    data = np.load(npz_path, allow_pickle=True)
    descriptors = data["descriptors"].astype(np.float32)
    image_names = data["image_names"].tolist()
    image_paths = data["image_paths"].tolist()
    return descriptors, image_names, image_paths


def load_descriptor_config(npz_path: Path) -> DescriptorConfig:
    """Load the descriptor configuration from the associated config.json.

    This function assumes that a 'config.json' file exists in the same
    directory as the provided descriptor database path.

    Args:
        npz_path (Path): Path to the 'descriptors.npz' database file.

    Returns:
        DescriptorConfig: The configuration object used to create the database.

    Raises:
        FileNotFoundError: If 'config.json' is not found in the same
                           directory as the npz file.
    """
    config_path = npz_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json next to database: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return DescriptorConfig(**data["descriptor_config"])


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(path: Path, payload: dict) -> None:
    """Save a dictionary to a JSON file with pretty-printing.

    This utility ensures the parent directory exists before writing the file.

    Args:
        path (Path): The output file path.
        payload (dict): The dictionary to serialize and save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NpEncoder)


def save_graph(path: Path, graph: nx.Graph | nx.DiGraph) -> None:
    """Serialize a networkx graph to a JSON file using node-link format.

    Args:
        path (Path): The output file path.
        graph (nx.Graph | nx.DiGraph): The graph object to save.
    """
    data = json_graph.node_link_data(graph)
    save_json(path, data)


def write_build_outputs(output_dir: Path, descriptors: np.ndarray, image_names: list[str], image_paths: list[str], descriptor_config: DescriptorConfig,
                        graph: nx.Graph, nav_graph: Optional[nx.DiGraph], action_edge_count: int,) -> None:
    """Save all artifacts from the database build process to disk.

    This function writes the following files to the specified output directory:
    - `descriptors.npz`: An archive containing the descriptor array and image
                         metadata.
    - `config.json`: A file with the descriptor configuration and summary stats.
    - `place_graph.json`: The structure of the place recognition graph.
    - `navigation_graph.json`: (Optional) The directed action graph.

    Args:
        output_dir (Path): The directory to save the artifacts to.
        descriptors (np.ndarray): The array of computed descriptors.
        image_names (list[str]): The list of image base names.
        image_paths (list[str]): The list of resolved image paths.
        descriptor_config (DescriptorConfig): The configuration used for
                                              descriptor extraction.
        graph (nx.Graph): The constructed place graph.
        nav_graph (Optional[nx.DiGraph]): The (optional) constructed nav graph.
        action_edge_count (int): The number of action edges added.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "descriptors.npz", descriptors=descriptors, image_names=np.array(image_names), image_paths=np.array(image_paths),)

    save_json(
        output_dir / "config.json",
        {
            "descriptor_config": asdict(descriptor_config),
            "artifact_files": {
                "descriptor_archive": "descriptors.npz",
                "place_graph": "place_graph.json",
                "navigation_graph": "navigation_graph.json" if nav_graph is not None else None,
            },
            "counts": {
                "num_images": len(image_paths),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_action_edges": action_edge_count,
            },
        },
    )

    save_graph(output_dir / "place_graph.json", graph)
    if nav_graph is not None:
        save_graph(output_dir / "navigation_graph.json", nav_graph)


def load_gray(path: Path) -> np.ndarray:
    """Load an image from a file and convert it to grayscale.

    Args:
        path (Path): Path to the image file.

    Returns:
        np.ndarray: The grayscale image as a NumPy array.

    Raises:
        RuntimeError: If `cv2.imread` fails to load the image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image



############ CosPlace Model & Descriptors

def get_device() -> torch.device:
    """
    Get the available torch device (CUDA if available, else CPU).
    Returns:
        torch.device: The device to use for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cosplace_model(repo_path: Optional[Path], config: DescriptorConfig, device: torch.device,
                        ) -> torch.nn.Module:
    """Load a trained CosPlace model for place recognition.

    This function abstracts the model loading process, supporting both local
    and remote (torch.hub) sources for the CosPlace repository.

    Args:
        repo_path (Optional[Path]): If provided, path to a local CosPlace repo.
                                    If None, the model is loaded from the remote
                                    torch.hub repository.
        config (DescriptorConfig): Configuration for the model's backbone and
                                   output dimension.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to which the
                               model should be loaded.

    Returns:
        torch.nn.Module: The loaded and initialized CosPlace model, set to
                         evaluation mode.

    Raises:
        FileNotFoundError: If a local `repo_path` is given but does not exist.
    """
    kwargs = {"backbone": config.backbone, "fc_output_dim": config.fc_output_dim, "trust_repo": True,}
    model: torch.nn.Module
    if repo_path is not None:
        repo_path = repo_path.expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"CosPlace repo not found: {repo_path}")
        model = torch.hub.load(str(repo_path), "get_trained_model", source="local", **kwargs) # type: ignore
    else:
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", **kwargs) # type: ignore

    model = model.eval().to(device)
    return model


def make_cosplace_transform(config: DescriptorConfig) -> T.Compose:
    """Create the standard image transformation pipeline for CosPlace.

    This pipeline resizes an image to the expected input dimensions, converts
    it to a PyTorch tensor, and normalizes its pixel values.

    Args:
        config (DescriptorConfig): Configuration specifying the target resize
                                   height and width.

    Returns:
        T.Compose: A torchvision transform pipeline that can be applied to a
                   PIL image.
    """
    return T.Compose([T.Resize((config.resize_height, config.resize_width)),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
                      )


def crop_image(img: Image.Image, crop_top_ratio: float, crop_bottom_ratio: float) -> Image.Image:
    """Crop an image vertically based on normalized ratios.

    Args:
        img (Image.Image): The input PIL image.
        crop_top_ratio (float): The normalized (0.0 to 1.0) vertical position
                                to start the crop.
        crop_bottom_ratio (float): The normalized (0.0 to 1.0) vertical position
                                   to end the crop.

    Returns:
        Image.Image: The cropped PIL image.

    Raises:
        ValueError: If the crop ratios are invalid (e.g., top >= bottom).
    """
    if not (0.0 <= crop_top_ratio < crop_bottom_ratio <= 1.0):
        raise ValueError("crop ratios must satisfy 0.0 <= top < bottom <= 1.0")
    width, height = img.size
    top = int(height * crop_top_ratio)
    bottom = int(height * crop_bottom_ratio)
    return img.crop((0, top, width, bottom))


def preprocess_image(path: Path, transform: T.Compose, config: DescriptorConfig) -> torch.Tensor:
    """Load, crop, and transform a single image from a file.

    This function encapsulates the full preprocessing pipeline for one image,
    making it ready for descriptor extraction. The output is a tensor with a
    leading batch dimension of 1.

    Args:
        path (Path): The path to the image file.
        transform (T.Compose): The torchvision transform pipeline to apply.
        config (DescriptorConfig): Configuration for cropping parameters.

    Returns:
        torch.Tensor: A 4D tensor of shape (1, C, H, W) ready for the model.
    """
    img = Image.open(path).convert("RGB")
    img = crop_image(img, config.crop_top_ratio, config.crop_bottom_ratio)
    tensor = transform(img)
    return tensor.unsqueeze(0)


def compute_descriptors(model: torch.nn.Module, image_paths: list[Path], config: DescriptorConfig,
                        device: torch.device) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute CosPlace descriptors for a list of images in batches.

    This function iterates through the provided image paths, preprocesses them,
    and feeds them through the model in batches to generate descriptors. It
    also collects metadata (image names and resolved paths).

    Args:
        model (torch.nn.Module): The trained CosPlace model.
        image_paths (list[Path]): A list of paths to the input images.
        config (DescriptorConfig): Configuration specifying batch size and
                                   preprocessing parameters.
        device (torch.device): The device on which to perform computation.

    Returns:
        tuple[np.ndarray, list[str], list[str]]: A tuple containing:
            - A NumPy array of descriptors, shape (N, D).
            - A list of image base names, length N.
            - A list of resolved absolute image path strings, length N.
    """
    transform = make_cosplace_transform(config)
    descs_list: list[np.ndarray] = []
    image_names: list[str] = []
    image_path_strings: list[str] = []

    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    for path in tqdm(image_paths, desc="Extracting descriptors"):
        tensor = preprocess_image(path, transform, config)
        batch_tensors.append(tensor)
        batch_paths.append(path)

        if len(batch_tensors) == config.batch_size:
            descs, names, path_strings = flush_descriptor_batch(model, batch_tensors, batch_paths, device)
            descs_list.append(descs)
            image_names.extend(names)
            image_path_strings.extend(path_strings)
            batch_tensors, batch_paths = [], []

    if batch_tensors:
        descs, names, path_strings = flush_descriptor_batch(model, batch_tensors, batch_paths, device)
        descs_list.append(descs)
        image_names.extend(names)
        image_path_strings.extend(path_strings)

    descriptors = np.concatenate(descs_list, axis=0)
    return descriptors, image_names, image_path_strings


def flush_descriptor_batch(model: torch.nn.Module, batch_tensors: list[torch.Tensor], batch_paths: list[Path],
                           device: torch.device) -> tuple[np.ndarray, list[str], list[str]]:
    """Process a single batch of tensors to compute descriptors.

    This is a helper for `compute_descriptors`. It takes a batch of
    preprocessed image tensors, computes their features, normalizes them, and
    returns the descriptors along with corresponding image metadata.

    Args:
        model (torch.nn.Module): The trained CosPlace model.
        batch_tensors (list[torch.Tensor]): A list of preprocessed image tensors.
        batch_paths (list[Path]): A list of paths corresponding to the tensors.
        device (torch.device): The device for computation.

    Returns:
        tuple[np.ndarray, list[str], list[str]]: A tuple containing:
            - A NumPy array of descriptors for the batch.
            - A list of image base names for the batch.
            - A list of resolved absolute image path strings for the batch.
    """
    batch = torch.cat(batch_tensors, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        feats = model(batch)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    descs = feats.cpu().numpy().astype(np.float32)
    names = [path.name for path in batch_paths]
    path_strings = [str(path.resolve()) for path in batch_paths]
    return descs, names, path_strings


def descriptor_distance_search(descriptors: np.ndarray, query_descriptor: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Find the top-K nearest neighbors in the descriptor database.

    This function computes the L2 (Euclidean) distance from a single query
    descriptor to all descriptors in the database and returns the indices and
    distances of the closest matches.

    Args:
        descriptors (np.ndarray): The database of descriptors (N, D).
        query_descriptor (np.ndarray): The query descriptor (D,).
        top_k (int): The number of nearest neighbors to retrieve.

    Returns:
        list[tuple[int, float]]: A list of the top-K `(index, distance)`
                                 tuples, sorted by distance.
    """
    dists = np.linalg.norm(descriptors - query_descriptor[None, :], axis=1)
    order = np.argsort(dists)[:top_k]
    return [(int(index), float(dists[index])) for index in order]



######################### Graph Building

def build_place_graph(descriptors: np.ndarray, image_names: list[str], image_paths: list[str], knn: int,) -> nx.Graph:
    """Build a graph connecting images by sequence and descriptor similarity.

    This function creates a graph where each node represents an image. It adds
    two types of edges:
    1. Sequence edges: Connecting consecutive images (e.g., node i to i+1).
    2. CosPlace edges: Connecting images that are among the top-K nearest
       neighbors in the descriptor space.

    Node attributes include 'index', 'name', and 'path'. Edge attributes
    distinguish between 'sequence' and 'cosplace' connections and store the
    descriptor distance ('desc_dist').

    Args:
        descriptors (np.ndarray): Array of descriptors for all images.
        image_names (list[str]): List of image base names.
        image_paths (list[str]): List of resolved image paths.
        knn (int): Number of nearest neighbors to connect for each node based
                   on descriptor similarity.

    Returns:
        nx.Graph: The constructed place recognition graph.
    """
    num_nodes = descriptors.shape[0]
    graph = nx.Graph()

    for index, (name, path) in enumerate(zip(image_names, image_paths)):
        graph.add_node(index, index=index, name=name, path=path)

    for index in range(num_nodes - 1):
        graph.add_edge(index, index + 1, sequence=True, cosplace=False, seq_weight=1.0)

    nn = NearestNeighbors(n_neighbors=min(knn + 1, num_nodes), metric="euclidean")
    nn.fit(descriptors)
    distances, indices = nn.kneighbors(descriptors)

    for i in range(num_nodes):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            if graph.has_edge(i, j):
                graph[i][j]["cosplace"] = True
                previous = graph[i][j].get("desc_dist", float("inf"))
                graph[i][j]["desc_dist"] = float(min(previous, dist))
            else:
                graph.add_edge(i, j, sequence=False, cosplace=True, desc_dist=float(dist))

    return graph


def build_step_image_action_maps(data_info: list[dict]) -> tuple[dict[int, str], dict[int, list[str]], dict[str, list[int]]]:
    """Create lookup maps from the structured data_info.

    This function processes the list of trajectory data to build three
    dictionaries for efficient lookups:
    - Mapping from a step number to the corresponding image name.
    - Mapping from a step number to the list of actions taken at that step.
    - Mapping from an image name to all step numbers where it appears.

    Args:
        data_info (list[dict]): A list of dictionaries, typically loaded from
                                data_info.json.

    Returns:
        tuple[dict[int, str], dict[int, list[str]], dict[str, list[int]]]:
            A tuple containing (step_to_image, step_to_action, image_to_steps).
    """
    step_to_image: dict[int, str] = {}
    step_to_action: dict[int, list[str]] = {}
    image_to_steps: dict[str, list[int]] = {}

    for entry in data_info:
        step = int(entry["step"])
        image_name = entry["image"]
        actions = entry.get("action", [])
        step_to_image[step] = image_name
        step_to_action[step] = actions
        image_to_steps.setdefault(image_name, []).append(step)

    return step_to_image, step_to_action, image_to_steps


def build_image_to_node_map(image_names: list[str]) -> dict[str, int]:
    """Create a mapping from image names to graph node indices.

    Args:
        image_names (list[str]): A list of image base names, where the list
                                 index corresponds to the node index in the graph.

    Returns:
        dict[str, int]: A dictionary mapping each image name to its integer
                        node index.
    """
    return {name: index for index, name in enumerate(image_names)}


def build_action_edges_from_json(step_to_image: dict[int, str], step_to_action: dict[int, list[str]], image_to_node: dict[str, int],
                                 ) -> list[tuple[int, int, list[str]]]:
    """Generate a list of action-annotated edges from trajectory data.

    This function iterates through consecutive steps in the trajectory. If an
    action was recorded for a step, it creates an edge from the node representing
    the current step's image to the node for the next step's image, annotating
    the edge with the list of actions.

    Args:
        step_to_image (dict[int, str]): Map from step number to image name.
        step_to_action (dict[int, list[str]]): Map from step number to actions.
        image_to_node (dict[str, int]): Map from image name to graph node index.

    Returns:
        list[tuple[int, int, list[str]]]: A list of tuples, where each
            represents an edge: (source_node_idx, target_node_idx, action_list).
    """
    action_edges: list[tuple[int, int, list[str]]] = []
    steps = sorted(step_to_image)
    retained_steps = [step for step in steps if image_to_node.get(step_to_image.get(step, "")) is not None]

    for current_step, next_step in zip(retained_steps[:-1], retained_steps[1:]):
        image_current = step_to_image.get(current_step)
        image_next = step_to_image.get(next_step)
        if image_current is None or image_next is None:
            continue

        node_u = image_to_node.get(image_current)
        node_v = image_to_node.get(image_next)
        if node_u is None or node_v is None:
            continue

        actions: list[str] = []
        for step in steps:
            if current_step <= step < next_step:
                actions.extend(step_to_action.get(step, []))

        if not actions:
            continue

        action_edges.append((node_u, node_v, actions))

    return action_edges


def attach_actions_to_graph(graph: nx.Graph, action_edges: list[tuple[int, int, list[str]]]) -> nx.DiGraph:
    """Create a directed navigation graph with action-annotated edges.

    This function initializes a new directed graph (`DiGraph`) with the same
    nodes as the input place graph. It then adds directed edges based on the
    `action_edges` list, annotating them with the corresponding actions. This
    resulting graph connects places with the actions required to move between
    them.

    Args:
        graph (nx.Graph): The base place recognition graph.
        action_edges (list[tuple[int, int, list[str]]]): A list of action-
            annotated edges to add to the new graph.

    Returns:
        nx.DiGraph: A new directed graph where edges represent traversable
                    paths and are labeled with actions.
    """
    nav_graph = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        nav_graph.add_node(node, **attrs)

    for u, v, actions in action_edges:
        if nav_graph.has_edge(u, v):
            merged = sorted(set(nav_graph[u][v].get("actions", [])) | set(actions))
            nav_graph[u][v]["actions"] = merged
        else:
            nav_graph.add_edge(u, v, actions=list(actions), from_json=True)

    return nav_graph


def resolve_superglue_module_dir(superglue_root: Path) -> Path:
    """Resolve the directory containing the SuperGlue/SuperPoint modules.

    The SuperGlue repository can be structured in two ways. This function
    checks if a 'models' subdirectory exists, which is the structure of the
    official repo. If so, it returns that path. Otherwise, it assumes the
    provided root is the correct directory to add to the Python path.

    Args:
        superglue_root (Path): Path to the root of the SuperGlue repository
                               or a directory containing its models.

    Returns:
        Path: The resolved path to be added to `sys.path`.

    Raises:
        FileNotFoundError: If the provided `superglue_root` does not exist.
    """
    superglue_root = superglue_root.expanduser().resolve()
    if not superglue_root.exists():
        raise FileNotFoundError(f"SuperGlue path not found: {superglue_root}")
    if (superglue_root / "models").is_dir():
        return superglue_root / "models"
    return superglue_root


def load_superglue_models(superglue_root: Path, device: torch.device):
    """Dynamically load SuperPoint and SuperGlue models for feature matching.

    This function temporarily adds the SuperGlue model directory to the Python
    path to allow for local imports. It then loads the pre-trained SuperPoint
    feature extractor and the indoor-tuned SuperGlue matcher.

    Args:
        superglue_root (Path): The root directory of the SuperGlue repository
                               or a directory with its models.
        device (torch.device): The device to load the models onto.

    Returns:
        tuple: A tuple containing the loaded (superpoint, superglue) models,
               both in evaluation mode.
    """
    models_dir = resolve_superglue_module_dir(superglue_root)
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))

    from superpoint import SuperPoint  # type: ignore
    from superglue import SuperGlue  # type: ignore

    superpoint = SuperPoint(
        {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024,
        }
    ).eval().to(device)
    superglue = SuperGlue(
        {
            "weights": "indoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        }
    ).eval().to(device)
    return superpoint, superglue


def match_superglue(image_a: np.ndarray, image_b: np.ndarray, superpoint, superglue, device: torch.device,) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Perform feature matching between two grayscale images using SuperGlue.

    This function takes two images, detects keypoints and descriptors using
    SuperPoint, and then finds matches between them using SuperGlue.

    Args:
        image_a (np.ndarray): The first grayscale image.
        image_b (np.ndarray): The second grayscale image.
        superpoint: The loaded SuperPoint model.
        superglue: The loaded SuperGlue model.
        device (torch.device): The device for computation.

    Returns:
        tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing
            two arrays of matched keypoints (one for each image). Returns
            (None, None) if fewer than 8 matches are found.
    """
    tensor_a = torch.from_numpy(image_a).float().div(255.0)[None, None].to(device)
    tensor_b = torch.from_numpy(image_b).float().div(255.0)[None, None].to(device)

    with torch.no_grad():
        features_a = superpoint({"image": tensor_a})
        features_b = superpoint({"image": tensor_b})
        data = {
            "image0": tensor_a,
            "image1": tensor_b,
            "keypoints0": features_a["keypoints"][0][None],
            "keypoints1": features_b["keypoints"][0][None],
            "scores0": features_a["scores"][0][None],
            "scores1": features_b["scores"][0][None],
            "descriptors0": features_a["descriptors"][0][None],
            "descriptors1": features_b["descriptors"][0][None],
        }
        prediction = superglue(data)

    matches0 = prediction["matches0"][0].cpu().numpy()
    keypoints0 = features_a["keypoints"][0].cpu().numpy()
    keypoints1 = features_b["keypoints"][0].cpu().numpy()
    valid = matches0 > -1
    if int(valid.sum()) < 8:
        return None, None

    matched0 = keypoints0[valid].astype(np.float32)
    matched1 = keypoints1[matches0[valid]].astype(np.float32)
    return matched0, matched1


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Construct a 3x3 camera intrinsic matrix from parameters.

    Args:
        fx (float): Focal length in the x-direction.
        fy (float): Focal length in the y-direction.
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.

    Returns:
        np.ndarray: The 3x3 intrinsic camera matrix.
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def estimate_motion(pts1: Optional[np.ndarray], pts2: Optional[np.ndarray], intrinsics: np.ndarray):
    """Estimate camera motion from matched keypoints.

    This function computes the essential matrix from two sets of matched points
    and then recovers the relative rotation and translation. It requires at
    least 8 point correspondences.

    Args:
        pts1 (Optional[np.ndarray]): Matched keypoints from the first image.
        pts2 (Optional[np.ndarray]): Matched keypoints from the second image.
        intrinsics (np.ndarray): The 3x3 camera intrinsic matrix.

    Returns:
        tuple: A tuple containing the (rotation, translation, pose_mask).
               Returns (None, None, None) if motion cannot be estimated.
    """
    if pts1 is None or pts2 is None or len(pts1) < 8:
        return None, None, None

    essential, _ = cv2.findEssentialMat(
        pts1,
        pts2,
        intrinsics,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None:
        return None, None, None
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, pts1, pts2, intrinsics)
    return rotation, translation, pose_mask


def verify_candidate(query_image: Path, candidate_image: Path, superpoint, superglue, device: torch.device,
                     intrinsics: np.ndarray, inlier_threshold: int,) -> dict:
    """Perform full geometric verification between a query and a candidate image.

    This function orchestrates the entire verification pipeline:
    1. Loads the two images in grayscale.
    2. Finds keypoint matches using `match_superglue`.
    3. Estimates relative motion using `estimate_motion`.
    4. Counts the number of inliers consistent with the motion model.
    5. Determines verification status based on an inlier threshold.

    Args:
        query_image (Path): Path to the query image.
        candidate_image (Path): Path to the database candidate image.
        superpoint: The loaded SuperPoint model.
        superglue: The loaded SuperGlue model.
        device (torch.device): The device for computation.
        intrinsics (np.ndarray): The camera intrinsic matrix.
        inlier_threshold (int): The minimum number of inliers required to
                                consider the match as verified.

    Returns:
        dict: A dictionary containing the number of 'inliers' and a boolean
              'verified' status.
    """
    image_a = load_gray(query_image)
    image_b = load_gray(candidate_image)
    pts1, pts2 = match_superglue(image_a, image_b, superpoint, superglue, device)
    _, _, pose_mask = estimate_motion(pts1, pts2, intrinsics)
    inliers = int(pose_mask.sum()) if pose_mask is not None else 0
    return {
        "inliers": inliers,
        "verified": inliers >= inlier_threshold,
    }


##################### Core Application Logic


def run_build_db(args: argparse.Namespace) -> None:
    """Execute the 'build-db' command workflow.

    This function orchestrates the entire process of building the baseline
    database artifacts. It reads configuration from the command-line
    arguments to:
    1. Collect and subsample image paths.
    2. Load the CosPlace model.
    3. Compute descriptors for all images.
    4. Build the place-recognition graph.
    5. (Optional) Build the action-annotated navigation graph.
    6. Write all artifacts to the output directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments for the
                                   'build-db' command.
    """
    device = get_device()
    descriptor_config = DescriptorConfig(backbone=args.backbone, fc_output_dim=args.descriptor_dim, resize_height=args.resize_height, resize_width=args.resize_width,
                                         crop_top_ratio=args.crop_top, crop_bottom_ratio=args.crop_bottom, batch_size=args.batch_size,)

    image_paths = collect_image_paths(args.image_dir, step=args.step, limit=args.limit)
    model = load_cosplace_model(args.cosplace_repo, descriptor_config, device)
    descriptors, image_names, image_path_strings = compute_descriptors(model, image_paths, descriptor_config, device)
    graph = build_place_graph(descriptors, image_names, image_path_strings, knn=args.knn)

    nav_graph: Optional[nx.DiGraph] = None
    action_edge_count = 0
    if args.data_info_json is not None:
        data_info = load_data_info(args.data_info_json)
        step_to_image, step_to_action, _ = build_step_image_action_maps(data_info)
        image_to_node = build_image_to_node_map(image_names)
        action_edges = build_action_edges_from_json(step_to_image, step_to_action, image_to_node)
        nav_graph = attach_actions_to_graph(graph, action_edges)
        action_edge_count = len(action_edges)

    write_build_outputs(output_dir=args.output_dir, descriptors=descriptors, image_names=image_names, image_paths=image_path_strings, descriptor_config=descriptor_config,
                        graph=graph, nav_graph=nav_graph, action_edge_count=action_edge_count,)

    print(f"Built baseline artifacts in {args.output_dir}")
    print(f"Images: {len(image_path_strings)}")
    print(f"Place graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    if nav_graph is not None:
        print(f"Navigation graph nodes: {nav_graph.number_of_nodes()}, edges: {nav_graph.number_of_edges()}")


def run_query(args: argparse.Namespace) -> None:
    """Execute the 'query' command workflow.

    This function orchestrates the process of localizing a single query image
    against a pre-built database. It reads configuration from the command-line
    arguments to:
    1. Load the descriptor database, config, and CosPlace model.
    2. Preprocess the query image and compute its descriptor.
    3. Perform a nearest neighbor search to find top-K candidates.
    4. (Optional) Perform geometric verification on candidates using SuperGlue.
    5. Print the results and optionally save them to a JSON file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments for the
                                   'query' command.
    """
    device = get_device()
    descriptor_config = load_descriptor_config(args.database)
    model = load_cosplace_model(args.cosplace_repo, descriptor_config, device)
    descriptors, image_names, image_paths = load_descriptor_archive(args.database)

    transform = make_cosplace_transform(descriptor_config)
    query_tensor = preprocess_image(args.query_image, transform, descriptor_config).to(device)
    with torch.no_grad():
        query_desc = model(query_tensor)
        query_desc = torch.nn.functional.normalize(query_desc, p=2, dim=1)
    query_desc_np = query_desc.cpu().numpy()[0].astype(np.float32)

    candidates = descriptor_distance_search(descriptors, query_desc_np, top_k=args.top_k)

    results: list[dict] = []
    superpoint = None
    superglue = None
    intrinsics = None
    if args.superglue_root is not None:
        superpoint, superglue = load_superglue_models(args.superglue_root, device)
        intrinsics = camera_matrix(args.fx, args.fy, args.cx, args.cy)

    for index, distance in candidates:
        result = {
            "index": index,
            "distance": distance,
            "image_name": image_names[index],
            "image_path": image_paths[index],
        }
        if superpoint is not None and superglue is not None and intrinsics is not None:
            verification = verify_candidate(
                query_image=args.query_image,
                candidate_image=Path(image_paths[index]),
                superpoint=superpoint,
                superglue=superglue,
                device=device,
                intrinsics=intrinsics,
                inlier_threshold=args.inlier_threshold,
            )
            result.update(verification)
        results.append(result)

    for rank, result in enumerate(results, start=1):
        line = f"[{rank}] idx={result['index']} dist={result['distance']:.4f} name={result['image_name']}"
        if "inliers" in result:
            line += f" inliers={result['inliers']} verified={result['verified']}"
        print(line)

    if args.results_json is not None:
        save_json(
            args.results_json,
            {
                "query_image": str(args.query_image.resolve()),
                "results": results,
            },
        )
        print(f"Saved query results to {args.results_json}")


def main() -> int:
    """Main entry point for the script.

    Parses command-line arguments and delegates to the appropriate subcommand
    handler (`run_build_db` or `run_query`).

    Returns:
        int: Exit code for the process.

    Raises:
        ValueError: If the parsed command is not supported.
    """
    args = parse_args()
    if args.command == "build-db":
        run_build_db(args)
        return 0
    if args.command == "query":
        run_query(args)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
