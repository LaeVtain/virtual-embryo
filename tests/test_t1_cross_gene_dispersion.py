import numpy as np
import pandas as pd
from scipy import sparse

from virtual_embryo.t1.cross_gene_dispersion import adjust_cross_gene_dispersion


def test_disjoint_gene_program_preserves_expression_marginals_and_support():
    rng = np.random.default_rng(222)
    latent = rng.gamma(2, size=(96, 4))
    loadings = rng.gamma(2, size=(4, 180))
    counts = rng.poisson(latent @ loadings / 12)
    values = np.log1p(10000 * counts / counts.sum(axis=1, keepdims=True))
    data = sparse.csr_matrix(values, dtype=np.float32)
    genes = pd.Index([f"Gene{i}" for i in range(data.shape[1])])
    changed = adjust_cross_gene_dispersion(data, genes)
    np.testing.assert_array_equal(changed.indices, data.indices)
    np.testing.assert_array_equal(changed.indptr, data.indptr)
    np.testing.assert_allclose(changed.toarray().astype(float).mean(axis=0),
                               data.toarray().astype(float).mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(np.expm1(changed.toarray().astype(float)).sum(axis=1),
                               np.expm1(data.toarray().astype(float)).sum(axis=1), rtol=1e-6)
    assert np.max(np.abs(changed.data - data.data)) > .01
