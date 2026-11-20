import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from virtual_embryo.t1.lineage_forecast import forecast
from virtual_embryo.t1.lineage_response import MARKERS


def test_lineage_forecast_uses_earlier_reference_stages_and_preserves_library_mass():
    genes = pd.Index([gene for group in MARKERS.values() for gene in group])
    rng = np.random.default_rng(17)
    counts = rng.poisson(0.3, size=(128, len(genes))).astype(float)
    for row in range(len(counts)):
        markers = MARKERS["myocardium" if row % 2 else "vascular"]
        counts[row, genes.get_indexer(markers)] += 3
    cells = sparse.csr_matrix(np.log1p(10000 * counts / counts.sum(1)[:, None]), dtype=np.float32)
    samples = np.repeat(["a", "b"], 64)
    stages = ["wt_7.75", "wt_8.25", "10x_8.5", "10x_16.5"]
    obs = pd.DataFrame({"cohort": np.repeat(stages, 128), "sample": np.tile(samples, 4)},
                       index=[f"ref_{i}" for i in range(512)])
    reference = ad.AnnData(sparse.vstack([cells] * 4, format="csr"), obs=obs, var=pd.DataFrame(index=genes))
    args = (cells[:64], genes, reference, cells, samples, cells, samples, np.ones(len(genes)), 8.5, 9.5, 101)
    with threadpool_limits(limits=1):
        first = forecast(*args)
        repeated = forecast(*args)
        nuclear_args = list(args)
        nuclear_args[4] = np.repeat("GSM8394736", cells.shape[0])
        nuclei = forecast(*nuclear_args, reference_assay="nucleus")
        nuclei_repeat = forecast(*nuclear_args, reference_assay="nucleus")
    assert (first != repeated).nnz == 0
    assert (nuclei != nuclei_repeat).nnz == 0
    assert (nuclei != first).nnz > 0
    assert first.shape == (64, len(genes))
    assert np.isfinite(first.data).all() and np.all(first.data > 0)
    mass = first.astype(np.float64)
    mass.data = np.expm1(mass.data)
    np.testing.assert_allclose(np.asarray(mass.sum(1)).ravel(), 10000, rtol=1e-6)
