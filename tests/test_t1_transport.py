import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from virtual_embryo.t1.transport import read_stage, transport


def test_selecting_named_genes_discards_stored_zero(tmp_path):
    x = sparse.csr_matrix(([2., 0., 3.], ([0, 0, 1], [0, 1, 1])), shape=(2, 2))
    path = tmp_path / "source.h5ad"
    ad.AnnData(x, var=pd.DataFrame(index=["a", "b"])).write_h5ad(path)
    matrix, panel = read_stage(path, pd.Index(["b", "a"]))
    assert list(panel) == ["b", "a"]
    assert matrix.nnz == 2
    np.testing.assert_array_equal(matrix.toarray(), [[0, 2], [3, 0]])


def test_transport_keeps_cell_support_and_library_size():
    linear = np.array([[2000, 0, 8000], [6000, 4000, 0], [0, 3000, 7000]])
    source = sparse.csr_matrix(np.log1p(linear))
    neural = np.log1p(np.roll(linear, 1, axis=1))
    shifted = transport(source, source, source, neural, horizon=2).toarray()
    np.testing.assert_array_equal(shifted > 0, linear > 0)
    np.testing.assert_allclose(np.expm1(shifted).sum(axis=1), 10000, rtol=1e-6)
    assert not np.allclose(shifted,source.toarray())
