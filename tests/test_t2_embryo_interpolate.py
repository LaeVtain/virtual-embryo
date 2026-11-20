import numpy as np
import pytest

from virtual_embryo.t2.embryo_interpolate import interpolate, registered_prediction
from virtual_embryo.t2.embryo_compartments import assign_donors, tissue_groups
from virtual_embryo.t2.tissue_registration import preserve_section_axis
from virtual_embryo.t2.section_volume import adjust_radius, tissue_volume


def test_bracketed_prediction_uses_observed_profiles_and_aligns_genes():
    rng = np.random.default_rng(421)
    genes = np.asarray(["A", "B", "C", "D", "E"])
    stages = {}
    for age in (6.75, 7.25, 8.0):
        stages[age] = dict(x=rng.uniform(0, 3, (120, 5)).astype(np.float32),
                           c=rng.normal(size=(120, 3)), labels=np.tile(["A", "B", "C"], 40),
                           genes=genes)
    expected = interpolate(stages, 7.5, 64, seed=11)
    assert expected["c"].shape == (64, 3)
    assert np.isfinite(expected["c"]).all()
    valid = set(map(tuple, np.vstack([stages[7.25]["x"], stages[8.0]["x"]])))
    assert all(tuple(row) in valid for row in expected["x"])
    stages[8.0]["x"] = stages[8.0]["x"][:, ::-1]
    stages[8.0]["genes"] = genes[::-1]
    result = interpolate(stages, 7.5, 64, seed=11)
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
    with pytest.raises(ValueError, match="exclude the target"):
        interpolate(stages, 7.25, 64)


def test_registration_preserves_donors_and_orders_narrow_gene_panels():
    rng = np.random.default_rng(913)
    genes = np.array(["X", "Y", "Z"])
    stages = {age: dict(x=rng.uniform(0, 2, (90, 3)).astype(np.float32), c=rng.normal(size=(90, 3)),
                        labels=np.repeat(["A", "B", "C"], 30), genes=genes)
              for age in (6.75, 7.25, 8.)}
    sampled = interpolate(stages, 7.5, 64, 211, include_donor=True)
    expected = registered_prediction(stages, 7.5, 64, 211)
    np.testing.assert_array_equal(sampled["x"], expected["x"])
    for later, age in ((False, 7.25), (True, 8.)):
        mask = sampled["later"] == later
        np.testing.assert_array_equal(expected["x"][mask], stages[age]["x"][sampled["row"][mask]])
    assert np.isfinite(expected["c"]).all()
    assert np.max(np.abs(expected["c"] - sampled["c"])) > .001
    stages[8.] = dict(stages[8.], x=stages[8.]["x"][:, ::-1], genes=genes[::-1])
    reordered = registered_prediction(stages, 7.5, 64, 211)
    np.testing.assert_array_equal(reordered["genes"], expected["genes"])
    np.testing.assert_array_equal(reordered["x"], expected["x"])
    np.testing.assert_allclose(reordered["c"], expected["c"], atol=1e-6)


def test_tissue_assignment_preserves_clock_and_reaches_feasible_counts():
    left = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    right = 1 - left
    sampled = np.array([False, False, False, False, True, True, True, True])
    result = assign_donors(left, right, sampled, np.array([.25, .75]))
    assert result.sum() == sampled.sum()
    np.testing.assert_array_equal(np.bincount(np.where(result, right, left)), [2, 6])


def test_incompatible_single_label_annotations_can_share_an_uninformative_group():
    rng = np.random.default_rng(41)
    stages = {age: dict(x=np.ones((9, 2), dtype=np.float32), c=rng.normal(size=(9, 3)),
                        labels=np.repeat(name, 9), genes=["A", "B"])
              for age, name in ((7.25, "early"), (8., "late"))}
    groups = tissue_groups(stages, 7.25, 8.)
    np.testing.assert_array_equal(groups[7.25], groups[8.])


def test_section_depth_keeps_its_identity_when_growth_changes_axis_order():
    grid = np.array(np.meshgrid([-1., 1.], [-1., 1.], [-1., 0., 1.])).reshape(3, -1).T
    earlier = np.repeat(grid * [1, 2, 6], 3, axis=0)
    later = np.repeat(grid * [6, 5, 1], 3, axis=0)
    predicted = (earlier + later).astype(np.float32)
    np.testing.assert_array_equal(preserve_section_axis(predicted, earlier, earlier*2, .5), predicted)
    corrected = preserve_section_axis(predicted, earlier, later, .5)
    assert np.max(np.abs(predicted-corrected)) > .1
    np.testing.assert_allclose(np.linalg.det(np.cov(corrected.T)), np.linalg.det(np.cov(predicted.T)), rtol=1e-5)


def test_section_volume_tracks_growth_and_preserves_the_predicted_shape():
    grid = np.array(np.meshgrid(np.arange(5), np.arange(6), np.arange(4))).reshape(3, -1).T.astype(float)
    stages = {age: {"c": grid * scale} for age, scale in ((0., 1.), (1., 2.), (2., 4.))}
    assert tissue_volume(stages[1.]["c"]) == pytest.approx(8 * tissue_volume(grid))
    predicted = grid + [7, 2, -4]
    actual = adjust_radius(stages, 1.5, predicted)
    expected = (predicted-predicted.mean(0))*np.sqrt(8) + predicted.mean(0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
