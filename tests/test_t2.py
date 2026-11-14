import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch
from threadpoolctl import threadpool_limits

from virtual_embryo.t2.calibrate import calibrate
from virtual_embryo.t2.networks import ConditionalEndpoint, ContinuousNeighborhoodField
from virtual_embryo.t2.predict import Predictor, mean_preserving_map
from virtual_embryo.t2.prepare import prepare
from virtual_embryo.t2.spatial import support_indices
from virtual_embryo.t2.train import train


@pytest.fixture
def model_inputs(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(77)
    torch.manual_seed(77)
    folder = tmp_path / "model"
    folder.mkdir()
    genes = np.array([f"gene{i}" for i in range(32)])
    config = {"heart": dict(times=[8.25, 8.5, 8.75, 9.5], coordinate_unit=100.,
                            spatial=dict(anchor_count=128, anchor_k=15, seed=20261111))}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "metadata.json").write_text(json.dumps({"T2_heart": {"genes": genes.tolist()}}))
    expression = ConditionalEndpoint(len(genes))
    spatial = ContinuousNeighborhoodField(len(genes) + 2)
    with torch.no_grad():
        expression.out.weight.normal_(0, .02)
        spatial.radial[-1].weight.normal_(0, .02)
        spatial.edge[-1].weight.normal_(0, .02)
    torch.save(dict(expression=expression.state_dict(), spatial=spatial.state_dict()), folder / "heart_control.pt")
    torch.save(dict(spatial=spatial.state_dict()), folder / "heart_spatial_continued.pt")
    knots = np.repeat(np.linspace(-3, 5, 257, dtype="float32")[:, None], len(genes), axis=1)
    joblib.dump({pair: (knots, np.maximum(knots, 0)) for pair in ((0, 1), (1, 2), (0, 2))}, folder / "heart_maps.joblib")
    paths = []
    for stage in range(3):
        x = rng.gamma(2, 1, size=(128, len(genes))).astype("float32")
        x[rng.random(x.shape) < .3] = 0
        c = rng.normal(0, 100, size=(128, 3)).astype("float32")
        path = tmp_path / f"stage{stage}.npz"
        np.savez_compressed(path, x=x, c=c, rows=np.arange(128), genes=genes)
        paths.append(path)
    return folder, paths


def test_gene_alignment_preserves_prediction_and_rejects_missing_panel(model_inputs):
    folder, paths = model_inputs
    model = Predictor("heart", model_dir=folder)
    with np.load(paths[0]) as source:
        x, c, genes = source["x"], source["c"], source["genes"]
    expected = model.predict(x, c, genes=genes, source_time=8.25, target_time=8.5)
    reordered = model.predict(x[:, ::-1].copy(), c, genes=genes[::-1], source_time=8.25, target_time=8.5)
    for key in ("x", "c", "genes"):
        assert_array_equal(expected[key], reordered[key])
    with pytest.raises(ValueError, match="Missing required genes"):
        model.predict(x[:, :-1], c, genes=genes[:-1], source_time=8.25, target_time=8.5)


def test_baseline_runs_without_calibration_assets(model_inputs):
    folder, paths = model_inputs
    (folder / "heart_maps.joblib").unlink()
    model = Predictor("heart", baseline=True, model_dir=folder)
    with np.load(paths[0]) as source:
        result = model.predict(source["x"], source["c"], genes=source["genes"], source_time=8.25, target_time=8.5)
    assert np.isfinite(result["x"]).all()
    assert (result["x"] >= 0).all()


def test_duplicate_coordinate_neighbors_exclude_self_and_collapsed_field_is_finite():
    positions = torch.zeros(20, 3, requires_grad=True)
    support = support_indices(positions.detach(), 15)
    assert not (support == torch.arange(20)[:, None]).any()
    network = ContinuousNeighborhoodField(6)
    result = network(positions, torch.ones(20, 6))
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))
    result.square().sum().backward()
    assert torch.isfinite(positions.grad).all()


def test_future_mapping_retains_mean_when_map_has_zero_mass():
    z = np.array([[1., -2.], [2., 3.], [4., 0.]], dtype="float32")
    p = np.array([[-3., -3.], [5., 5.]], dtype="float32")
    target = np.zeros_like(p)
    result = mean_preserving_map(z, (p, target))
    assert_array_equal(result, np.maximum(z, 0))
    assert_allclose(result.mean(0), np.maximum(z, 0).mean(0))


def test_prepare_train_calibrate_predict_pipeline(model_inputs, tmp_path):
    folder, paths = model_inputs
    prepared = tmp_path / "prepared.joblib"
    with threadpool_limits(limits=1):
        prepare("heart", [[path] for path in paths], prepared, folder)
        control = tmp_path / "control.pt"
        train("heart", prepared, control, mode="control", steps=2, model_dir=folder)
        checkpoint = torch.load(control, weights_only=True)
        assert checkpoint["updates"] == 2
        assert checkpoint["expression"]["out.weight"].abs().sum() > 0
        torch.save(checkpoint, folder / "heart_control.pt")
        first, second = tmp_path / "continue1.pt", tmp_path / "continue2.pt"
        train("heart", prepared, first, steps=2, model_dir=folder)
        train("heart", prepared, second, steps=2, model_dir=folder)
        a, b = torch.load(first, weights_only=True), torch.load(second, weights_only=True)
        for group in ("expression", "spatial"):
            for name, value in a[group].items():
                assert torch.equal(value, b[group][name])
                assert torch.isfinite(value).all()
        torch.save({"spatial": a["spatial"]}, folder / "heart_spatial_continued.pt")
        maps = calibrate("heart", prepared, folder / "heart_maps.joblib", folder)
        for source, target in maps.values():
            assert source.shape == target.shape == (257, 32)
            assert (np.diff(source, axis=0) >= 0).all()
            assert (np.diff(target, axis=0) >= 0).all()
    output = tmp_path / "prediction.npz"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-m", "virtual_embryo.t2.predict", "--setting", "heart", "--model-dir", str(folder),
                    "--input", str(paths[0]), "--output", str(output), "--source-time", "8.25", "--target-time", "8.5"],
                   env=env, check=True, capture_output=True, text=True)
    with np.load(output) as result:
        assert result["x"].shape == (128, 32)
        assert result["c"].shape == (128, 3)
        assert np.isfinite(result["x"]).all()
        assert np.isfinite(result["c"]).all()
