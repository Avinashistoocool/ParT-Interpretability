import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from partinterp import analyses  # noqa: E402
from partinterp.data import FEAT, synthetic_store  # noqa: E402
from partinterp.models import ParTKinAdapter, ParTAdapter  # noqa: E402
from run_robustness import corrupt_features, knockout_mask  # noqa: E402


def test_threshold_and_matched_knockout():
    store = synthetic_store(n_jets=12, max_particles=24, seed=4)
    idx = np.arange(len(store))
    low = knockout_mask(store, idx, 24, 2, False)
    high = knockout_mask(store, idx, 24, 5, False)
    assert low.sum() >= high.sum()
    assert (low & ~store.mask).sum() == 0
    matched = knockout_mask(store, idx, 24, 3, True)
    assert not (matched & knockout_mask(store, idx, 24, 3, False)).any()
    assert (matched.sum(1) <= knockout_mask(store, idx, 24, 3, False).sum(1)).all()


def test_corruptions_touch_only_selected_features():
    store = synthetic_store(n_jets=8, max_particles=20, seed=8)
    idx = np.arange(len(store))
    feats, _, _ = ParTAdapter.inputs(store, idx, 20, "cpu")
    flags = analyses.particle_physics(store, idx, 20)
    mask = torch.as_tensor(flags["displaced"])
    for method, channels in (("zero_d0", [FEAT["part_d0"]]),
                             ("zero_d0_dz", [FEAT["part_d0"], FEAT["part_dz"]]),
                             ("prompt_mean", [FEAT["part_d0"], FEAT["part_dz"]])):
        altered = corrupt_features(feats, flags, method)
        other = [i for i in range(feats.shape[1]) if i not in channels]
        torch.testing.assert_close(altered[:, other], feats[:, other])
        torch.testing.assert_close(altered[:, channels].transpose(1, 2)[~mask],
                                   feats[:, channels].transpose(1, 2)[~mask])
        if method.startswith("zero"):
            assert (altered[:, channels].transpose(1, 2)[mask] == 0).all()


def test_kin_adapter_uses_exact_official_feature_order():
    store = synthetic_store(n_jets=3, max_particles=12)
    idx = np.arange(3)
    full, v, mk = ParTAdapter.inputs(store, idx, 12, "cpu")
    kin, vk, mkk = ParTKinAdapter.inputs(store, idx, 12, "cpu")
    torch.testing.assert_close(kin, full[:, [0, 1, 2, 3, 4, 15, 16]])
    torch.testing.assert_close(v, vk)
    torch.testing.assert_close(mk, mkk)


def test_prompt_mean_leaves_jets_without_prompt_reference_unchanged():
    store = synthetic_store(n_jets=2, max_particles=12)
    feats, _, _ = ParTAdapter.inputs(store, np.arange(2), 12, "cpu")
    flags = {"charged": np.zeros((2, 12), dtype=bool), "displaced": np.ones((2, 12), dtype=bool)}
    torch.testing.assert_close(corrupt_features(feats, flags, "prompt_mean"), feats)
