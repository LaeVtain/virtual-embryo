import anndata as ad
import numpy as np
import pytest
from common import core_metrics as core, shape_metrics as shape

from virtual_embryo.evaluate import score
from virtual_embryo.evaluate import evaluation_populations
from virtual_embryo.export import export_h5ad


@pytest.fixture
def populations():
    rng = np.random.default_rng(21)
    genes = np.asarray([f"g{i}" for i in range(40)])
    reference = {"x": rng.poisson(8, (64, 40)).astype(float), "genes": genes}
    truth = {"x": rng.poisson(np.r_[np.full(20, 20), np.full(20, 2)], (64, 40)).astype(float),
             "c": rng.normal(size=(64, 3)), "genes": genes}
    prediction = {**truth, "x": truth["x"] * 0.9}
    return prediction, truth, reference


def test_evaluation_aligns_genes_and_covers_ranking_metrics(populations):
    pred, truth, ref = populations
    permuted = {**pred, "x": pred["x"][:, ::-1], "genes": pred["genes"][::-1]}
    result = score("T2", permuted, truth, ref)
    assert len(result["metrics"]) == 8
    assert result["metrics"]["de_score"] == core.de_score(pred["x"], truth["x"], ref["x"])["score"]
    assert result["metrics"]["occupancy_dice"] == shape.occupancy_dice(pred["c"], truth["c"])[0]
    assert result["metrics"]["scale_log_ratio"] == pytest.approx(0.0)
    t3 = score("T3", pred, truth, ref)
    assert t3["metrics"]["severity_slope"] == core.severity_slope(pred["x"], truth["x"], ref["x"])[0]


def test_prediction_export_preserves_values_and_gene_order(populations, tmp_path):
    pred = populations[0]
    source = tmp_path / "prediction.npz"
    np.savez_compressed(source, **pred)
    output = export_h5ad(source, tmp_path / "prediction.h5ad")
    result = ad.read_h5ad(output)
    np.testing.assert_array_equal(result.X.toarray(), pred["x"].astype(np.float32))
    np.testing.assert_array_equal(result.obsm["spatial_3D"], pred["c"].astype(np.float32))
    assert result.var_names.tolist() == pred["genes"].tolist()


def test_capped_evaluation_fixes_truth_and_reference_across_candidates(populations, tmp_path):
    pred, truth, ref = populations
    paths = []
    for name, pop in [("prediction", pred), ("truth", truth), ("reference", ref)]:
        path = tmp_path / (name + ".npz")
        np.savez(path, **pop)
        paths.append(path)
    short = tmp_path / "short.npz"
    np.savez(short, x=pred["x"][:15], genes=pred["genes"], c=pred["c"][:15])
    comparison = evaluation_populations(paths, None, 24, 41)
    smaller = evaluation_populations([short, *paths[1:]], None, 24, 41)
    copy_last = evaluation_populations([paths[2], *paths[1:]], None, 24, 41)
    for other in (smaller, copy_last):
        for position in (1, 2):
            np.testing.assert_array_equal(other[position]["x"], comparison[position]["x"])
    np.testing.assert_array_equal(copy_last[0]["x"], copy_last[2]["x"])
