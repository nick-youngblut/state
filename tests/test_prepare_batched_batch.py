import numpy as np
import pytest
import torch

from state._cli._tx._infer import prepare_batched_batch


@pytest.mark.parametrize("num_perts", [1, 2, 16, 64])
def test_prepare_batched_batch_shapes_and_device(num_perts):
    win_size = 3
    input_dim = 5
    pert_dim = 4
    device = torch.device("cpu")

    ctrl_basal_list = [np.ones((win_size, input_dim), dtype=np.float32) * i for i in range(num_perts)]
    pert_vecs = [torch.full((pert_dim,), float(i)) for i in range(num_perts)]
    pert_names = [f"p{i}" for i in range(num_perts)]

    batch = prepare_batched_batch(
        ctrl_basal_list=ctrl_basal_list,
        pert_vecs=pert_vecs,
        batch_indices_list=None,
        pert_names=pert_names,
        win_size=win_size,
        device=device,
    )

    assert batch["ctrl_cell_emb"].shape == (num_perts * win_size, input_dim)
    assert batch["pert_emb"].shape == (num_perts * win_size, pert_dim)
    assert len(batch["pert_name"]) == num_perts * win_size
    expected_names = []
    for name in pert_names:
        expected_names.extend([name] * win_size)
    assert batch["pert_name"] == expected_names
    assert batch["ctrl_cell_emb"].device.type == "cpu"
    assert batch["pert_emb"].device.type == "cpu"


def test_prepare_batched_batch_with_batch_indices():
    win_size = 2
    input_dim = 3
    pert_dim = 2
    device = torch.device("cpu")

    ctrl_basal_list = [np.zeros((win_size, input_dim), dtype=np.float32) for _ in range(2)]
    pert_vecs = [torch.zeros(pert_dim), torch.ones(pert_dim)]
    pert_names = ["a", "b"]
    batch_indices_list = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([2, 3], dtype=torch.long)]

    batch = prepare_batched_batch(
        ctrl_basal_list=ctrl_basal_list,
        pert_vecs=pert_vecs,
        batch_indices_list=batch_indices_list,
        pert_names=pert_names,
        win_size=win_size,
        device=device,
    )

    assert "batch" in batch
    assert batch["batch"].shape == (4,)
    assert batch["batch"].tolist() == [0, 1, 2, 3]
