import numpy as np
import pandas as pd
from scipy import sparse

from virtual_embryo.t1.state_dispersion import adjust_state_dispersion


def test_state_dispersion_preserves_support_gene_means_and_cell_totals():
    genes = pd.Index(["Hba-a1", "Hbb-y", "Myl7", "Tnnt2", "Kdr", "Krt8", "Dcn", "Wt1"])
    rng = np.random.default_rng(29)
    counts = rng.poisson(4, size=(80, len(genes)))
    counts[:40, :2] += 7
    counts[40:, 2:4] += 8
    counts[rng.random(counts.shape) < 0.2] = 0
    values = np.log1p(10000 * counts / counts.sum(axis=1, keepdims=True))
    source = sparse.csr_matrix(values, dtype=np.float32)
    changed = adjust_state_dispersion(source, genes)
    np.testing.assert_array_equal(changed.indices, source.indices)
    np.testing.assert_array_equal(changed.indptr, source.indptr)
    np.testing.assert_allclose(changed.toarray().astype(float).mean(axis=0),
                               source.toarray().astype(float).mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(np.expm1(changed.toarray().astype(float)).sum(axis=1),
                               np.expm1(source.toarray().astype(float)).sum(axis=1), rtol=1e-6)
    assert np.max(np.abs(source.data - changed.data)) > 0.02
