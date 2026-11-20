import numpy as np

from virtual_embryo.t2.embryo_capture import aligned_stage_means


def test_overlap_weights_remove_tissue_and_depth_composition_change():
    genes = np.array(["a", "b"])
    stages = {}
    for age, counts in [(6.75, (90, 30, 30)), (7.25, (30, 90, 30)), (8.0, (30, 30, 90))]:
        category = np.repeat(np.arange(3), counts)
        x = np.array([[1., 0.], [0., 2.], [3., 0.]], dtype=np.float32)[category]
        stages[age] = dict(x=x, c=np.zeros((len(x), 3)), genes=genes,
                           labels=np.array(["a", "a", "b"])[category],
                           depth=np.array([8., 64., 64.])[category])
    measured = aligned_stage_means(stages, genes)
    np.testing.assert_allclose(measured, [[4/3, 2/3]] * 3)
    assert not np.allclose(stages[6.75]["x"].mean(0), stages[8.]["x"].mean(0))
