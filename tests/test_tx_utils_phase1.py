import numpy as np
import pytest
import torch

from state.tx.utils.batch_utils import chunk, estimate_max_pert_batch_size, pregenerate_all_control_indices
from state.tx.utils.compile_utils import maybe_compile_model


class DummyModel(torch.nn.Module):
    def __init__(self, pert_dim: int):
        super().__init__()
        self.pert_dim = pert_dim
        self.linear = torch.nn.Linear(2, 2)


def test_maybe_compile_model_disabled():
    model = torch.nn.Linear(2, 2)
    compiled = maybe_compile_model(model, enabled=False)
    assert compiled is model


def test_maybe_compile_model_cpu(monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    compiled = maybe_compile_model(model, enabled=True, quiet=True)
    assert compiled is model


def test_maybe_compile_model_failure(monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "__version__", "2.1.0", raising=False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "compile", _boom, raising=False)
    with pytest.warns(UserWarning):
        compiled = maybe_compile_model(model, enabled=True, quiet=True)
    assert compiled is model


def test_estimate_max_pert_batch_size_reasonable(monkeypatch):
    model = DummyModel(pert_dim=10)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000, 2_000_000))

    batch_size = estimate_max_pert_batch_size(
        model=model,
        cell_set_len=4,
        hidden_dim=8,
        input_dim=6,
        output_dim=5,
        num_layers=2,
        safety_factor=1.0,
    )

    input_mem = 4 * (6 + 10) * 4
    attention_mem = 4 * 4 * 2 * 4
    activation_mem = 4 * 8 * 4 * 2 * 4
    output_mem = 4 * 5 * 4
    mem_per_pert = input_mem + attention_mem + activation_mem + output_mem
    expected = max(1, int(1_000_000 / mem_per_pert))

    assert batch_size == expected


def test_estimate_max_pert_batch_size_cpu(monkeypatch):
    model = DummyModel(pert_dim=10)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert estimate_max_pert_batch_size(model, 4, 8, 6, 5, 2) == 1


def test_chunk_partitions_list():
    items = list(range(5))
    chunks = list(chunk(items, 2))
    assert chunks == [[0, 1], [2, 3], [4]]


def test_pregenerate_all_control_indices_matches_sequential():
    unique_groups = np.array(["A", "B"], dtype=object)
    group_labels = np.array(["A", "A", "B", "B", "B"], dtype=object)
    pert_names_all = np.array(["p1", "p2", "p1", "p1", "p2"], dtype=object)

    control_map = {
        "A": np.array([10, 11], dtype=int),
        "B": np.array([20, 21, 22], dtype=int),
    }

    def group_control_indices(group):
        return control_map[group]

    rng = np.random.RandomState(0)
    batched = pregenerate_all_control_indices(
        rng=rng,
        unique_groups=unique_groups,
        group_labels=group_labels,
        pert_names_all=pert_names_all,
        group_control_indices_fn=group_control_indices,
        cell_set_len=2,
    )

    rng_expected = np.random.RandomState(0)
    expected = {}
    for group in unique_groups:
        grp_idx = np.where(group_labels == group)[0]
        grp_ctrl_pool = group_control_indices(group)
        unique_perts = np.unique(pert_names_all[grp_idx])
        for pert in unique_perts:
            pert_idx = grp_idx[pert_names_all[grp_idx] == pert]
            windows = []
            start = 0
            while start < len(pert_idx):
                win_size = min(2, len(pert_idx) - start)
                ctrl_indices = rng_expected.choice(grp_ctrl_pool, size=win_size, replace=True)
                windows.append(ctrl_indices)
                start += win_size
            expected[(group, pert)] = windows

    assert batched.keys() == expected.keys()
    for key, windows in expected.items():
        assert len(batched[key]) == len(windows)
        for left, right in zip(batched[key], windows):
            assert np.array_equal(left, right)
