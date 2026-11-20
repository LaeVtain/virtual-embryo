import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from virtual_embryo.t2.temporal import predict_population, read_stage


def observations():
    rng = np.random.default_rng(17)
    genes = np.array(["B", "A", "C", "D"])
    left = {"x": rng.integers(0, 4, size=(96, 4)).astype("float32"),
            "c": rng.normal(size=(96, 3)), "genes": genes}
    right = {"x": left["x"] * (left["x"] > 1), "c": 0.5 * left["c"], "genes": genes}
    return left, right


def test_long_horizon_keeps_growth_bounded_and_library_size():
    old, new = observations()
    result = predict_population(old, new, 8.25, 8.5, 10.5, len(new["x"]))
    radius = lambda coords: np.sqrt(np.square(coords - coords.mean(0)).sum(1).mean())
    assert 0.5 < radius(result["c"]) / radius(new["c"]) < 1
    np.testing.assert_allclose(np.sort(np.expm1(result["x"]).sum(1)),
                               np.sort(np.expm1(new["x"]).sum(1)), rtol=1e-5)


def test_bracket_interpolation_aligns_panel_and_preserves_cell_profiles():
    old, new = observations()
    result = predict_population(old, new, 7.5, 8, 7.75, 65)
    reordered = predict_population(old, dict(new, genes=new["genes"][::-1], x=new["x"][:, ::-1]),
                                   7.5, 8, 7.75, 65)
    for key in result:
        np.testing.assert_array_equal(result[key], reordered[key])
    profiles = set(map(tuple, np.vstack([old["x"], new["x"]])))
    assert all(tuple(row) in profiles for row in result["x"])
    with pytest.raises(ValueError, match="target"):
        predict_population(old, new, 7.5, 8, 8, 65)


@pytest.mark.parametrize("csr", [False, True])
def test_read_official_dense_and_sparse_stage(tmp_path, csr):
    pop, _ = observations()
    x = sparse.csr_matrix(pop["x"]) if csr else pop["x"]
    data = ad.AnnData(x, var=pd.DataFrame(index=pop["genes"]))
    data.obsm["spatial_3D"] = pop["c"]
    path = tmp_path / "stage.h5ad"
    data.write_h5ad(path)
    result = read_stage(path, ["D", "B", "A"])
    np.testing.assert_array_equal(result["x"], pop["x"][:, [3, 0, 1]])
    np.testing.assert_array_equal(result["c"], pop["c"])
