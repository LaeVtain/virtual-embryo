import numpy as np
import pytest

from virtual_embryo.t2.heart_state_bridge import transfer


def test_branch_uses_native_panel_order_and_preserves_libraries():
    rng = np.random.default_rng(391)
    n, genes = 100, np.array([f"g{i}" for i in range(24)])
    counts = rng.poisson(.6, (n, len(genes)))
    counts[:, 0] = 1
    depth = counts.sum(axis=1)
    x = np.log1p(10000 * counts / depth[:, None]).astype(np.float32)
    coordinates = rng.normal(size=(n, 3)).astype(np.float32)
    labels = np.full(n, "V-CM")
    present = np.ones(len(genes), dtype=bool)
    present[-1] = False
    model = {"genes": genes, "early_age": 9.5, "late_age": 14.5, "present": present,
             "axes": rng.normal(size=(3, present.sum())), "center": np.zeros(present.sum()),
             "myocardial_left": rng.normal(size=(2, 3)), "myocardial_right": rng.normal(size=(2, 3)),
             "myocardial_flow": np.array([[.3, .2], [.1, .4]]),
             "myocardial_p0": rng.uniform(.01, .3, (2, present.sum())),
             "myocardial_p1": rng.uniform(.08, .55, (2, present.sum()))}
    baseline = dict(x=x, c=coordinates, genes=genes)
    moved = transfer(baseline, x, labels, depth, (9.5, 10.5, 12.5), 101, model)
    flipped = dict(x=x[:, ::-1], c=coordinates, genes=genes[::-1])
    backwards = transfer(flipped, flipped["x"], labels, depth, (9.5, 10.5, 12.5), 101, model)
    np.testing.assert_array_equal(moved["x"], backwards["x"][:, ::-1])
    np.testing.assert_array_equal(moved["c"], coordinates)
    np.testing.assert_allclose(np.expm1(moved["x"]).sum(axis=1), 10000, rtol=2e-6)
    assert np.any((moved["x"] > 0) != (x > 0))
    assert transfer(baseline, x, labels, depth, (8.5, 9.5, 10.5), 101, model) is baseline
    with pytest.raises(ValueError, match="protected T2 heart"):
        transfer(baseline, x, labels, depth, (9.5, 10.5, 12.5), 101, {**model, "early_age": 8.5})
    estimated = {**model, "minimum_supported_days": 2.}
    assert transfer(baseline, x, labels, depth, (8.25, 8.75, 10.5), 101, estimated, "animal-horizon") is baseline
    full = transfer(baseline, x, labels, depth, (9.5, 10.5, 12.5), 101, model, "fixed")
    gated = transfer(baseline, x, labels, depth, (9.5, 10.5, 12.5), 101, estimated, "animal-horizon")
    np.testing.assert_array_equal(gated["x"], full["x"])
