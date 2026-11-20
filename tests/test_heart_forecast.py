import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from virtual_embryo.t2.heart_forecast import developmental_change, molecule_counts, predict


def test_cardiac_forecast_aligns_observed_genes_and_preserves_cell_libraries():
    rng = np.random.default_rng(733)
    genes = np.array([f"g{i}" for i in range(32)])
    stages = []
    for n in [90, 120]:
        counts = rng.poisson(1.5, size=(n, len(genes)))
        counts[:, 0] = 1
        x = np.log1p(10000 * counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
        np.testing.assert_array_equal(molecule_counts(x), counts)
        stages.append(dict(x=x, c=rng.normal(size=(n, 3)), genes=genes))
    trend = pd.DataFrame({"early": rng.normal(0, .1, len(genes)), "late": rng.normal(0, .1, len(genes))},
                         index=genes)
    earlier, later = stages
    reordered = dict(x=earlier["x"][:, ::-1], c=earlier["c"], genes=genes[::-1])
    with threadpool_limits(limits=1):
        original = predict(earlier, later, (9.5, 10.5, 12.5), 60, 58, trend)
        aligned = predict(reordered, later, (9.5, 10.5, 12.5), 60, 58, trend.iloc[::-1])
    np.testing.assert_array_equal(original["x"], aligned["x"])
    np.testing.assert_array_equal(original["c"], aligned["c"])
    np.testing.assert_allclose(np.expm1(original["x"]).sum(axis=1), 10000, rtol=2e-6)
    assert original["c"].shape == (60, 3)
    assert len(np.unique(original["c"], axis=0)) == 60


def test_earlier_mask_does_not_use_the_withheld_e95_ventricular_sample():
    genes = ["a", "b"]
    trend = pd.DataFrame({"early": [1., -1.], "late": [.1, -.2]}, index=genes)
    before = developmental_change(trend, genes, 8.75, 9.5)
    trend["early"] *= -1
    np.testing.assert_array_equal(before, developmental_change(trend, genes, 8.75, 9.5))


def test_spatial_niche_forecast_is_independent_of_storage_layout():
    rng = np.random.default_rng(331)
    genes = np.array([f"g{i}" for i in range(40)])
    stages = []
    for n in (200, 320):
        count = rng.poisson(1.1, (n, len(genes)))
        counts_log = np.log1p(10000 * count / count.sum(axis=1)[:, None]).astype(np.float32)
        stages.append({"x": counts_log, "c": rng.normal(size=(n, 3)), "genes": genes})
    model = {"genes": genes, "components": rng.normal(size=(5, len(genes))),
             "center": rng.normal(size=len(genes)), "centroids": rng.normal(size=(3, 5)),
             "early_mean": np.full(len(genes), 1 / len(genes)), "bulk": rng.normal(0, .3, len(genes)),
             "local": rng.normal(0, .2, (3, len(genes))), "mass": np.array([-.2, .1, .35])}
    rates = pd.DataFrame(rng.normal(0, .1, (len(genes), 2)), index=genes, columns=["early", "late"])
    with threadpool_limits(limits=1):
        one = predict(*stages, (8.75, 9.5, 10.5), 180, 77, rates, niches=model)
        column_major = [dict(stage, x=np.asfortranarray(stage["x"])) for stage in stages]
        two = predict(*column_major, (8.75, 9.5, 10.5), 180, 77, rates, niches=model)
    np.testing.assert_array_equal(one["x"], two["x"])
    np.testing.assert_array_equal(one["c"], two["c"])
