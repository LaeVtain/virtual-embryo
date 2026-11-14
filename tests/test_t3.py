import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch

from virtual_embryo.t3.predict import predict
from virtual_embryo.t3.runtime import features
from virtual_embryo.t3.train import calibrate, fit_mean, fit_state, histogram, prepare_projection, train


def training_data():
    rng = np.random.default_rng(71)
    x = np.log1p(rng.gamma(2, 2, (64, 12)))
    y = np.log1p(rng.gamma(3, 2, (64, 12)))
    genes = np.array(["Mab21l2", "Gata4"] + [f"g{i}" for i in range(10)])
    return x, y, genes


def initial_models(folder):
    folder.mkdir()
    x, y, genes = training_data()
    tilt = {"genes": genes}
    for condition in ("mab", "gata"):
        tilt.update(prepare_projection(x, genes, condition))
        model = calibrate(x, y, tilt, condition, np.linspace(-0.2, 0.2, 8))
        np.savez_compressed(folder / f"{condition}.npz", **model)
    np.savez_compressed(folder / "tilt.npz", **tilt)
    return x, y, genes, tilt


def test_histogram_uses_positive_midpoint_quantiles():
    x = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [8.0, 8.0]])
    rates, knots = histogram(x, np.ones(4))
    assert_array_equal(rates, [0.75, 1.0])
    assert_array_equal(knots[:, 0], [2.0, 2.0])
    assert_array_equal(knots[:, -1], [8.0, 8.0])
    assert_array_equal(knots[:, 32], [4.0, 5.0])


def test_frozen_calibration_preserves_state_and_other_condition(tmp_path):
    source = tmp_path / "initial"
    x, y, genes, tilt = initial_models(source)
    wt, ko = tmp_path / "wt.npz", tmp_path / "ko.npz"
    np.savez_compressed(wt, x=x, genes=genes)
    np.savez_compressed(ko, x=y[:, ::-1], genes=genes[::-1])
    output = tmp_path / "fitted"
    audit = train(wt, ko, "Mab21l2", output, source)
    assert audit["fit_state"] is False
    for filename in ("tilt.npz", "mab.npz", "gata.npz"):
        with np.load(source / filename) as before, np.load(output / filename) as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                assert_array_equal(before[key], after[key])


def test_prediction_gene_reordering_and_coordinate_sampling(tmp_path):
    model_dir = tmp_path / "model"
    x, _, genes, _ = initial_models(model_dir)
    coords = np.arange(len(x) * 3).reshape(-1, 3)
    result = predict(x, coords, genes, "mab", model_dir=model_dir)
    reordered = predict(x[:, ::-1], coords, genes[::-1], "Mab21l2", model_dir=model_dir)
    for key in result:
        assert_array_equal(result[key], reordered[key])
    assert_array_equal(result["c"], coords[result["source_indices"]])
    assert_allclose(np.expm1(result["x"]).sum(1), 10000, rtol=1e-13)
    assert_allclose(result["weights"].sum(), 1.0)


def test_mean8_adam_first_step_and_convergence():
    torch.set_num_threads(2)
    source = np.linspace(-1.0, 1.0, 64)[:, None]
    target = np.array([0.35])
    beta1, first = fit_mean(source, source, target, np.ones(1), 0.35**2, steps=1)
    beta, fitted = fit_mean(source, source, target, np.ones(1), 0.35**2, steps=200)
    assert_allclose(beta1, [0.03], atol=1e-8)
    assert first["trace"][0]["update"] == 1
    assert fitted["trace"][-1]["objective"] < 0.1 * first["trace"][0]["objective"]
    assert 0 < beta[0] < 4


def test_state8_training_excludes_perturbed_gene():
    x, y, genes = training_data()
    tilt, beta, audit = fit_state(x, y, genes, "mab", support_indices=np.arange(16), steps=2)
    assert not tilt["mab_keep"][0]
    assert tilt["mab_keep"].sum() == len(genes) - 1
    assert tilt["mab_components"].shape == (8, 11)
    assert_allclose(features(x, tilt, "mab").std(0), np.ones(8), rtol=1e-12)
    assert beta.shape == (8,)
    assert np.isfinite(beta).all()
    assert audit["support_indices"] == list(range(16))


def test_train_and_predict_cli(tmp_path):
    x, y, genes = training_data()
    wt, ko = tmp_path / "wt.npz", tmp_path / "ko.npz"
    np.savez_compressed(wt, x=x, c=np.zeros((len(x), 3)), genes=genes)
    np.savez_compressed(ko, x=y, genes=genes)
    model_dir = tmp_path / "model"
    prediction = tmp_path / "prediction.npz"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    fitted = subprocess.run([sys.executable, "-m", "virtual_embryo.t3.train", "--wt", str(wt), "--ko", str(ko),
                             "--condition", "mab", "--output-dir", str(model_dir), "--fit-state",
                             "--support-size", "16", "--steps", "2"], env=env, check=True, capture_output=True, text=True)
    assert json.loads(fitted.stdout)["fit_state"] is True
    subprocess.run([sys.executable, "-m", "virtual_embryo.t3.predict", "--input", str(wt), "--output", str(prediction),
                    "--condition", "mab", "--model-dir", str(model_dir)], env=env, check=True, capture_output=True, text=True)
    with np.load(prediction) as result:
        assert result["x"].shape == x.shape
        assert np.isfinite(result["x"]).all()
        assert_allclose(np.expm1(result["x"]).sum(1), 10000, rtol=1e-13)
