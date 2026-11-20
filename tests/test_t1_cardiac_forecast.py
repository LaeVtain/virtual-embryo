import numpy as np
from scipy import sparse

from virtual_embryo.t1.cardiac_forecast import forecast, molecule_response


def counts_to_log(counts):
    return sparse.csr_matrix(np.log1p(counts*10000/counts.sum(axis=1,keepdims=True)),dtype=np.float32)


def test_reconstructed_umis_preserve_observations_when_factors_are_one():
    counts = np.array([[1,2,0,11],[3,1,12,0],[1,6,0,3],[2,1,2,20]])
    matrix = counts_to_log(counts)
    result = molecule_response(matrix,np.ones(4),31)
    np.testing.assert_allclose(result.toarray(),matrix.toarray(),rtol=1e-6)


def test_atlas_forecast_is_seeded_and_library_normalized():
    rng = np.random.default_rng(49)
    source_counts = rng.poisson(1.5,size=(32,24))
    anchor_counts = rng.poisson(2.5,size=(48,24))
    source_counts[:,0] = 1
    anchor_counts[:,0] = 1
    donors = counts_to_log(source_counts)
    anchor = counts_to_log(anchor_counts)
    labels = np.repeat(["replicate_a","replicate_b"],24)
    factors = np.linspace(0.7,1.5,24)
    first = forecast(donors,anchor,labels,factors,0.5,101)
    second = forecast(donors,anchor,labels,factors,0.5,101)
    assert (first != second).nnz == 0
    assert first.shape == donors.shape
    assert np.isfinite(first.data).all() and np.all(first.data > 0)
    np.testing.assert_allclose(np.expm1(first.toarray()).sum(axis=1),10000,rtol=1e-6)
