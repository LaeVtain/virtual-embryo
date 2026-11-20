import numpy as np
import pandas as pd
from scipy import sparse

from virtual_embryo.t1.chamber_projection import project_chambers


def test_adjacent_reference_and_finite_cell_libraries(tmp_path):
    genes = pd.Index(["Tnnt2", "Myl7", "Cdh5", "Kdr", "Rplp0"])
    counts = np.tile([[12, 8, 1, 0, 5], [0, 1, 9, 7, 4]], (24, 1))
    x = sparse.csr_matrix(np.log1p(counts * (10000 / counts.sum(1))[:, None]), dtype=np.float32)
    path = tmp_path / "chambers.npz"
    np.savez(path, genes=genes.to_numpy(dtype=str), times=[9.5, 14.5],
             lineages=["myocardium", "vascular"], log_change=[[0.8,-0.4,0,0,0], [0,0,0.3,-0.2,0]],
             reliability=np.ones((2, 5)), composition_lineages=["myocardium", "vascular"],
             composition_log_change=[0.5, -0.5])
    assert project_chambers(x, x, genes, path, 8.5, 10.5) is x
    assert project_chambers(x, x, genes, path, 12.0, 12.5) is x
    y = project_chambers(x, x, genes, path, 10.5, 12.5)
    assert y.shape == x.shape and (y != x).nnz
    assert (y != project_chambers(x, x, genes, path, 10.5, 12.5)).nnz == 0
    normalized = y.astype(np.float64)
    normalized.data = np.expm1(normalized.data)
    np.testing.assert_allclose(normalized.sum(1), 10000, rtol=1e-6)
