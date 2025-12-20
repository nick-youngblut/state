from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Callable, Hashable, Iterator, TypeVar

import numpy as np
import torch
from torch import nn


T = TypeVar("T")


def get_gpu_free_memory() -> int | None:
    """Return free GPU memory in bytes, or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        free_mem, _ = torch.cuda.mem_get_info()
    except Exception as exc:
        warnings.warn(f"Could not query GPU free memory: {exc}")
        return None
    return int(free_mem)


def estimate_max_pert_batch_size(
    model: nn.Module,
    cell_set_len: int,
    hidden_dim: int,
    input_dim: int,
    output_dim: int,
    num_layers: int,
    safety_factor: float = 0.8,
) -> int:
    """Estimate the maximum perturbation batch size that fits in GPU memory."""
    if not torch.cuda.is_available():
        return 1

    free_memory = get_gpu_free_memory()
    if free_memory is None:
        return 1

    pert_dim = getattr(model, "pert_dim", 0) or 0

    input_mem = cell_set_len * (input_dim + pert_dim) * 4
    attention_mem = cell_set_len * cell_set_len * num_layers * 4
    activation_mem = cell_set_len * hidden_dim * 4 * num_layers * 4
    output_mem = cell_set_len * output_dim * 4

    mem_per_pert = input_mem + attention_mem + activation_mem + output_mem
    if mem_per_pert <= 0:
        return 1

    max_batch = int((free_memory * safety_factor) / mem_per_pert)
    return max(1, max_batch)


def chunk(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive chunks from an iterable."""
    if size <= 0:
        raise ValueError("chunk size must be positive")

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def pregenerate_all_control_indices(
    rng: np.random.RandomState,
    unique_groups: np.ndarray,
    group_labels: np.ndarray,
    pert_names_all: np.ndarray,
    group_control_indices_fn: Callable[[Hashable], np.ndarray],
    cell_set_len: int,
) -> dict[tuple[Hashable, Hashable], list[np.ndarray]]:
    """Pre-generate control indices in sequential RNG order for reproducibility."""
    result: dict[tuple[Hashable, Hashable], list[np.ndarray]] = {}

    for group in unique_groups:
        grp_idx = np.where(group_labels == group)[0]
        grp_ctrl_pool = group_control_indices_fn(group)
        unique_perts = np.unique(pert_names_all[grp_idx])

        for pert in unique_perts:
            pert_idx = grp_idx[pert_names_all[grp_idx] == pert]
            start = 0
            windows: list[np.ndarray] = []
            while start < len(pert_idx):
                win_size = min(cell_set_len, len(pert_idx) - start)
                ctrl_indices = rng.choice(grp_ctrl_pool, size=win_size, replace=True)
                windows.append(ctrl_indices)
                start += win_size
            result[(group, pert)] = windows

    return result
