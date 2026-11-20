import numpy as np

from virtual_embryo.t2.heart_sampling import donor_adjustment, stratified_counts


def test_gene_detection_matches_integrated_probability_with_rare_and_absent_markers():
    rng = np.random.default_rng(714)
    n = 103
    counts = rng.integers(0, 5, (n, 4))
    counts[:, 2] = 0
    counts[:, 3] = 1
    rates = np.array([.05, .43, .8, 1.])
    coords = rng.normal(size=(n, 3))
    got = stratified_counts(counts, rates, coords, seed=29)
    expected = (1 - (1 - rates) ** counts).sum(axis=0)
    np.testing.assert_array_less(np.abs((got > 0).sum(axis=0) - expected), 1.)
    np.testing.assert_array_equal(got, stratified_counts(counts, rates, coords, seed=29))
    np.testing.assert_array_equal(got[:, 2], 0)
    np.testing.assert_array_equal(got[:, 3], 1)


def test_full_population_has_no_donor_sampling_error():
    raw = np.array([[0, 1], [1, 2], [3, 1]])
    x = np.log1p(raw).astype(np.float32)
    rows = np.array([2, 0, 1])
    np.testing.assert_array_equal(donor_adjustment(x, np.array([.1, .3, .6]), rows, raw[rows]), [0., 0.])
