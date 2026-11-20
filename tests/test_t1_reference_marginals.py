import numpy as np
from scipy import sparse

from virtual_embryo.t1.reference_marginals import exchange_marginals


def test_nuclear_marginal_exchange_preserves_gene_detection_and_cell_mass():
    random = np.random.default_rng(2357)
    cell_states = random.gamma(1.8, size=(60, 3))
    loadings = random.gamma(1.3, size=(3, 170))

    def normalized(mu):
        count = random.poisson(mu)
        return sparse.csr_matrix(
            np.log1p(10000 * count / count.sum(axis=1, keepdims=True)), dtype=np.float32
        )

    whole = normalized(cell_states @ loadings / 8)
    nuclear = normalized(cell_states[::-1] @ loadings / 11)
    result = exchange_marginals(whole, nuclear)
    again = exchange_marginals(whole, nuclear)
    np.testing.assert_array_equal(result.toarray(), again.toarray())
    np.testing.assert_array_equal(np.bincount(result.indices, minlength=170),
                                  np.bincount(nuclear.indices, minlength=170))
    np.testing.assert_allclose(np.expm1(result.toarray().astype(float)).sum(1), 10000, rtol=1e-6)
    assert not np.array_equal(result.toarray(), nuclear.toarray())
