import numpy as np
import pandas as pd
from scipy import sparse

from virtual_embryo.t1.age_prior import adjust_cells, gene_multipliers


def test_trend_aligns_genes_and_preserves_library_and_support():
    genes = ["a", "b", "unmeasured"]
    trend = pd.Series({"b": np.log(0.25), "a": np.log(4.0)})
    multiplier = gene_multipliers(trend, genes, 1, 2)
    np.testing.assert_allclose(multiplier, [2, 0.5, 1])
    source = sparse.csr_matrix(np.log1p([[2000, 8000, 0], [1000, 0, 9000]]))
    result = adjust_cells(source, multiplier)
    linear = np.expm1(result.toarray())
    np.testing.assert_allclose(linear.sum(axis=1), 10000, rtol=1e-6)
    np.testing.assert_array_equal(linear > 0, source.toarray() > 0)
    np.testing.assert_allclose(linear[0, 0], linear[0, 1], rtol=1e-6)
