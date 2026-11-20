import numpy as np
import pytest
from virtual_embryo.t2.atlas_interpolate import predict_with_atlas
from virtual_embryo.t2.temporal import predict_population


def test_shared_gene_correction_excludes_held_age_and_preserves_libraries():
    rng = np.random.default_rng(73)
    genes = np.array(["A", "B", "C", "D", "E"])
    times = np.array([6.5, 6.75, 7., 7.25, 8.])
    stages = {}
    for t in (6.5, 6.75, 8.):
        stages[t] = dict(x=rng.uniform(0,1,(128,5)).astype(np.float32),
                         c=rng.normal(size=(128,3)), genes=genes)
    atlas = dict(times=times, genes=genes[:-1], stage_means=rng.uniform(.3,.8,(5,4)))
    pred = predict_with_atlas(stages,7.25,50,521,atlas)
    atlas["stage_means"][3] = 1e4
    repeated = predict_with_atlas(stages,7.25,50,521,atlas)
    prior = predict_population(stages[6.75],stages[8.],6.75,8.,7.25,50,521)
    np.testing.assert_array_equal(pred["x"],repeated["x"])
    np.testing.assert_array_equal(pred["c"],prior["c"])
    np.testing.assert_allclose(np.expm1(pred["x"]).sum(1),np.expm1(prior["x"]).sum(1),rtol=2e-6)
    with pytest.raises(ValueError, match="protected age"):
        predict_with_atlas(stages,7.25,50,521,dict(atlas,times=np.array([6.5,6.75,7.,7.5,8.])))
