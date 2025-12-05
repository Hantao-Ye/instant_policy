"""
Instant Policy: In-Context Imitation Learning via Graph Diffusion

This package provides the GraphDiffusion model for robotic manipulation tasks.
"""

from .instant_policy import GraphDiffusion, sample_to_cond_demo
from .utils import (
    downsample_pcd,
    pose_to_transform,
    transform_to_pose,
    transform_pcd,
    subsample_pcd,
    remove_statistical_outliers,
)

__version__ = "0.1.0"

__all__ = [
    "GraphDiffusion",
    "sample_to_cond_demo",
    "downsample_pcd",
    "pose_to_transform",
    "transform_to_pose",
    "transform_pcd",
    "subsample_pcd",
    "remove_statistical_outliers",
]
