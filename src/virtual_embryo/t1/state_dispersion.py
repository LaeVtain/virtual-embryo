"""Adjust joint RNA expression across continuous cell states while preserving gene marginals."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import rankdata

from .lineage_response import MARKERS


def state_coordinates(cells, genes):
    """Rank cells by library complexity and by the expression of each measured marker panel."""
    n = cells.shape[0]
    breadth = np.log1p(np.diff(cells.indptr))
    columns = [(rankdata(breadth) - 0.5) / n]
    for markers in MARKERS.values():
        positions = genes.get_indexer(markers)
        available = positions[positions >= 0]
        expression = (np.asarray(cells[:, available].mean(axis=1)).ravel()
                      if len(available) else np.zeros(n))
        columns.append((rankdata(expression) - 0.5) / n)
    coordinates = np.stack(columns, axis=1)
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.maximum(0.05, coordinates.std(axis=0))
    return coordinates


def restore_marginals(changed, original):
    """Correct the cell totals in the null space of the fixed gene-sum constraints."""
    result = changed.astype(np.float64, copy=True)
    lengths = np.diff(result.indptr)
    if np.any(lengths == 0):
        raise ValueError("Each predicted cell must contain a measured gene.")
    genes = result.shape[1]
    desired = np.bincount(original.indices, weights=original.data.astype(np.float64), minlength=genes)
    sums = np.bincount(result.indices, weights=result.data, minlength=genes)
    result.data *= np.divide(desired, sums, out=np.ones_like(desired), where=sums > 0)[result.indices]
    totals = np.add.reduceat(np.expm1(original.data.astype(np.float64)), original.indptr[:-1])
    for _ in range(25):
        values = result.data
        mass = np.expm1(values)
        residual = totals - np.add.reduceat(mass, result.indptr[:-1])
        error = np.max(np.abs(residual / totals))
        if error < 1e-7:
            return result.astype(np.float32)
        derivative = 1 + mass
        weights = (mass / derivative)**2
        column_weights = np.bincount(result.indices, weights=weights, minlength=genes)
        row_scale = 1 / np.sqrt(np.add.reduceat(mass**2, result.indptr[:-1]))
        column_scale = np.divide(1, np.sqrt(column_weights), out=np.zeros_like(column_weights),
                                 where=column_weights > 0)
        jacobian = result.copy()
        jacobian.data = weights * derivative * np.repeat(row_scale, lengths) * column_scale[result.indices]

        def multiply(vector):
            return vector - jacobian @ (jacobian.T @ vector) + 1e-10 * vector

        system = LinearOperator((len(totals), len(totals)), matvec=multiply)
        step, status = cg(system, row_scale * residual, rtol=1e-7, maxiter=90)
        if status < 0:
            raise ValueError("Could not solve the cell-total constraints.")
        row_step = np.repeat(row_scale * step, lengths)
        column_step = np.bincount(result.indices, weights=weights * derivative * row_step,
                                  minlength=genes) * column_scale**2
        update = weights * (derivative * row_step - column_step[result.indices])
        negative = update < 0
        fraction = min(1., .9 * np.min(values[negative] / -update[negative])) if negative.any() else 1.
        while fraction > 1e-7:
            proposed = values + fraction * update
            new_totals = np.add.reduceat(np.expm1(proposed), result.indptr[:-1])
            if np.max(np.abs(new_totals / totals - 1)) < error:
                break
            fraction *= .5
        result.data = values + fraction * update
    raise ValueError("The predicted cell totals did not converge while preserving the gene means.")


def adjust_state_dispersion(cells, genes):
    """Increase shared state variation without moving the detected genes in any cell."""
    cells = cells.tocsr()
    coordinates = state_coordinates(cells, genes)
    n = cells.shape[0]
    covariance = coordinates.T @ coordinates / n + 0.15 * np.eye(coordinates.shape[1])
    gene_means = np.asarray(cells.mean(axis=0, dtype=np.float64)).ravel()
    profile = np.asarray(cells.T @ coordinates).T / n
    response = np.linalg.solve(covariance, profile) / (0.15 + gene_means)
    changed = cells.astype(np.float64, copy=True)
    for start in range(0, n, 64):
        end = min(start + 64, n)
        conditional = coordinates[start:end] @ response
        left, right = changed.indptr[[start, end]]
        rows = np.repeat(np.arange(end - start), np.diff(changed.indptr[start:end + 1]))
        shift = 0.7 * np.clip(conditional[rows, changed.indices[left:right]], -1.2, 1.2)
        changed.data[left:right] = np.log1p(np.expm1(changed.data[left:right]) * np.exp(shift))
    return restore_marginals(changed, cells)
