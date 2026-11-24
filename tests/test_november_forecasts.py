import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from virtual_embryo.t1.reference_marginals import project_nuclear_means
from virtual_embryo.t2.heart_anatomy import refine_shape, rms_radius
from virtual_embryo.t2.local_expression import place_local_profiles
from virtual_embryo.t3.response_ranks import couple_response


def test_nuclear_forecast_follows_late_means_at_fixed_cell_support():
    counts = np.array([[7, 2, 1, 0], [6, 3, 1, 0], [8, 1, 1, 0]], dtype=float)
    source = sparse.csr_matrix(np.log1p(1000 * np.tile(counts, (20, 1))), dtype=np.float32)
    late = source[:, [2, 1, 0, 3]].tocsr()
    next_day = project_nuclear_means(source, source, late, 10.5, 11.5)
    three_days = project_nuclear_means(source, source, late, 10.5, 13.5)

    assert (source[:, 2].mean() < next_day[:, 2].mean() < three_days[:, 2].mean())
    assert (source[:, 0].mean() > next_day[:, 0].mean() > three_days[:, 0].mean())
    np.testing.assert_array_equal(three_days.indices, source.indices)
    np.testing.assert_array_equal(three_days.indptr, source.indptr)
    np.testing.assert_allclose(np.expm1(three_days.toarray().astype(float)).sum(1), 10000, rtol=1e-6)


def test_atlas_refinement_preserves_center_and_radius_at_different_horizons():
    rng = np.random.default_rng(104)
    source = rng.normal(size=(145, 3)) * [2, 5, 9] + [150, -70, 80]
    reference = rng.gamma(shape=1.4, size=(287, 3)) * [12, 6, 4] + [-90, 210, -120]
    short = refine_shape(source, 10.5, 11.0, reference)
    long = refine_shape(source, 10.5, 12.5, reference)

    np.testing.assert_array_equal(long, refine_shape(source, 10.5, 12.5, reference))
    assert np.linalg.norm(long - source) > np.linalg.norm(short - source) > 0
    np.testing.assert_allclose(long.mean(0, dtype=float), source.mean(0), rtol=0, atol=2e-5)
    np.testing.assert_allclose(rms_radius(long.astype(float)), rms_radius(source), rtol=2e-6)


def test_neighborhood_assignment_preserves_complete_cell_profiles_with_duplicates():
    rng = np.random.default_rng(412)
    expression = rng.normal(size=(120, 27)).astype(np.float32)
    expression[1] = expression[0]
    coordinates = rng.normal(size=(len(expression), 3)).astype(np.float32)
    with threadpool_limits(limits=1):
        placed = place_local_profiles(expression, coordinates)
        np.testing.assert_array_equal(placed, place_local_profiles(expression, coordinates))
    assert not np.array_equal(placed, expression)
    original, counts = np.unique(expression, axis=0, return_counts=True)
    assigned, multiplicity = np.unique(placed, axis=0, return_counts=True)
    np.testing.assert_array_equal(assigned, original)
    np.testing.assert_array_equal(multiplicity, counts)
    np.testing.assert_array_equal(place_local_profiles(expression[:1], coordinates[:1]), expression[:1])


def test_response_coupling_uses_wild_type_cell_identity_and_preserves_each_library():
    rng = np.random.default_rng(302)
    mutant = np.log1p(rng.poisson(.9, size=(130, 72))).astype(np.float32)
    wild_type = np.log1p(rng.poisson(1.4, size=mutant.shape)).astype(np.float32)
    with threadpool_limits(limits=1):
        coupled = couple_response(wild_type, mutant, 21)
        repeated = couple_response(wild_type, mutant, 21)
        permuted = couple_response(wild_type[::-1], mutant, 21)
    np.testing.assert_array_equal(coupled, repeated)
    assert np.count_nonzero(permuted != coupled) > .25 * mutant.size
    np.testing.assert_array_equal((coupled > 0).sum(0), (mutant > 0).sum(0))
    np.testing.assert_allclose(np.expm1(coupled.astype(float)).sum(1),
                               np.expm1(mutant.astype(float)).sum(1), rtol=2e-6)
