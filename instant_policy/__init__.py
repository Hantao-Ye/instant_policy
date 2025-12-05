"""
Instant Policy: In-Context Imitation Learning via Graph Diffusion

This package provides the GraphDiffusion model for robotic manipulation tasks.
"""

from .instant_policy import GraphDiffusion, sample_to_cond_demo

__version__ = "0.1.0"

__all__ = ["GraphDiffusion", "sample_to_cond_demo"]
