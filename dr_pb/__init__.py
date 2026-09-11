"""Minimal Sinkhorn-DRO performance-boosting controller implementation."""

from .config import ExperimentConfig
from .controller import PerfBoostController
from .disturbances import DisturbanceSpec, LatentDisturbanceDataset, ReferenceSampler
from .systems import RobotsSystem
from .losses import RobotsLoss
from .trainers import train_saa, train_dro, train_dro_lambda_grid
