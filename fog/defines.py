import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

class FogParams(BaseModel):
    model_dim: int = Field(default=320, description="Model dimension")
    block_size: int = Field(default=15552, description="Block size")
    patch_size: int = Field(default=18, description="Patch size")
    num_encoders: int = Field(default=2, description="# encoder layers")
    num_lstms: int = Field(default=2, description="# LSTM layers")
    num_heads: int = Field(default=2, description="# transformer heads")
    training: bool = Field(default=False, description="Training mode")
    dropout: float = Field(default=0, description="Dropout rate")


class TrainParams(BaseModel):
    job_dir: Path = Field(default_factory=tempfile.gettempdir, description="Job output directory")

    # Dataset params
    ds_path: Path = Field(default_factory=Path, description="Dataset directory")

    # Train params
    learning_rate = Field(1e-3, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    buffer_size: int = Field(default=2000, description="Buffer size")
    steps_per_epoch: Field(64, description="Steps per epoch")
    epochs: int = Field(50, description="Number of epochs")

    # Model params
    model_dim: int = Field(default=320, description="Model dimension")
    block_size: int = Field(default=15552, description="Block size")
    patch_size: int = Field(default=18, description="Patch size")
    num_encoders: int = Field(default=2, description="# encoder layers")
    num_lstms: int = Field(default=2, description="# LSTM layers")
    num_heads: int = Field(default=2, description="# transformer heads")
    dropout: float = Field(default=0, description="Dropout rate")
    weights_file: Path | None = Field(None, description="Path to a checkpoint weights to load")

    num_workers: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="Number of workers"
    )
