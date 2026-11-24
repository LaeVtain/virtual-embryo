import numpy as np

from virtual_embryo.t2.heart_capture import match_detection_counts
from virtual_embryo.t2.measured_neighborhoods import measured_neighborhoods


def test_detection_calibration_preserves_each_cells_molecules():
    rng = np.random.default_rng(57)
    counts = rng.poisson(.6, (96, 48)).astype(float)
    counts[:,0] += 1
    expression = np.log1p(counts).astype(np.float32)
    coordinates = rng.normal(size=(len(counts),3))
    native = (counts>0).sum(1)
    desired = native + np.where(np.arange(len(counts))%2==0,2,-2)
    predicted = match_detection_counts(expression,coordinates,desired,73)
    np.testing.assert_array_equal((predicted>0).sum(1),desired)
    np.testing.assert_allclose(np.expm1(predicted.astype(float)).sum(1),counts.sum(1),rtol=1e-6)


def test_measured_neighborhood_matching_preserves_entire_cell_profiles():
    rng=np.random.default_rng(979)
    native=rng.normal(size=(240,26)).astype(np.float32)
    points=rng.normal(size=(240,3))
    donors=rng.choice(240,128,replace=False)
    expression=native[donors].copy()
    parent=dict(x=native,c=points)
    revised=measured_neighborhoods(expression,points[donors],[parent],donors,np.zeros(len(donors),dtype=int),.65)
    before=expression[np.argsort(expression[:,0])]
    after=revised[np.argsort(revised[:,0])]
    np.testing.assert_array_equal(before,after)
