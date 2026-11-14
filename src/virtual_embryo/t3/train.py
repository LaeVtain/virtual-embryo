"""Fit state8 WT weights or calibrate KO quantiles with frozen state weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch

from ..paths import task_assets
from .runtime import CONDITIONS, GRID, PERTURBED_GENES, features


def histogram(x, weights):
    """Positive detection rates and 65 weighted midpoint quantiles."""
    weights = weights / weights.sum()
    rates = weights @ (x > 0)
    knots = np.zeros((x.shape[1], len(GRID)))
    for gene in range(x.shape[1]):
        positive = x[:, gene] > 0
        if not positive.any():
            continue
        order = np.argsort(x[positive, gene], kind="stable")
        values = x[positive, gene][order]
        w = weights[positive][order]
        knots[gene] = np.interp(GRID, (np.cumsum(w) - 0.5 * w) / w.sum(), values)
    return rates, knots


def calibrate(x, y, tilt, condition, coefficients):
    """Final n1024_frozen recipe, retaining only the parameters used at runtime."""
    # Match the original calibration's BLAS reduction order.
    with threadpool_limits(limits=1):
        weights = softmax(features(x, tilt, condition) @ coefficients)
        _, source_knots = histogram(x, weights)
        target_rate, target_knots = histogram(y, np.ones(len(y)))
    target_knots[target_rate == 0] = source_knots[target_rate == 0]
    return dict(target_rate=target_rate, target_knots=target_knots, coefficients=np.asarray(coefficients))


def prepare_projection(x, genes, condition):
    """PCA8 fitted to WT training cells, excluding the perturbed gene."""
    keep = np.asarray(genes).astype(str) != PERTURBED_GENES[condition]
    center = x[:, keep].mean(0)
    scale = np.maximum(x[:, keep].std(0), 0.1)
    standardized = (x[:, keep] - center) / scale
    pca = PCA(8, svd_solver="full").fit(standardized)
    pcscale = np.maximum(pca.transform(standardized).std(0), 1e-8)
    values = dict(keep=keep, center=center, scale=scale, components=pca.components_, pca_mean=pca.mean_, pcscale=pcscale)
    return {condition + "_" + key: value for key, value in values.items()}


def fit_mean(features, source, target_mean, gene_scale, base_scale, ridge=0.1, bound=4.0, steps=800):
    """Original mean8 normalized expression loss and Adam update schedule."""
    f = torch.as_tensor(features, dtype=torch.float32)
    x = torch.as_tensor(source, dtype=torch.float32)
    target = torch.as_tensor(target_mean, dtype=torch.float32)
    scale = torch.as_tensor(gene_scale, dtype=torch.float32)
    beta = torch.nn.Parameter(torch.zeros(f.shape[1]))
    optimizer = torch.optim.Adam([beta], lr=0.03)
    trace = []
    for step in range(steps):
        weights = torch.softmax(f @ beta, 0)
        mean_loss = (((weights @ x - target) / scale) ** 2).mean() / base_scale
        penalty = ridge / 2 * beta.square().sum()
        loss = mean_loss + penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_([beta], 10, error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-bound, bound)
        if step % 100 == 0 or step == steps - 1:
            trace.append(dict(update=step + 1, objective=float(loss.detach()), mean_term=float(mean_loss.detach()),
                              ridge_term=float(penalty.detach()), gradient_norm=gradient, coefficients=beta.detach().tolist()))
    return beta.detach().double().numpy(), dict(optimizer="Adam", steps=steps, lr=0.03, ridge=ridge, bound=bound, trace=trace)


def fit_state(x, y, genes, condition, support_indices=None, support_size=256, support_seed=1,
              steps=800, projection=None):
    """Train the original mean8 arm on one ordered WT/KO training split."""
    tilt = {"genes": np.asarray(genes).astype(str)}
    tilt.update(prepare_projection(x, genes, condition) if projection is None else projection)
    if support_indices is None:
        condition_index = {"mab": 0, "gata": 1}[condition]
        support_indices = np.random.default_rng(20261104 + 100 * support_seed + condition_index).choice(
            len(x), support_size, replace=False
        )
    support_indices = np.asarray(support_indices, dtype=np.int64)
    source = x[support_indices]
    target = y.mean(0)
    keep = tilt[condition + "_keep"]
    gene_scale = np.maximum(x.std(0), 0.1)
    base_scale = max(float(np.mean(((source.mean(0) - target) / gene_scale)[keep] ** 2)), 1e-6)
    coefficients, audit = fit_mean(features(source, tilt, condition), source[:, keep], target[keep],
                                   gene_scale.astype("float32")[keep], base_scale, steps=steps)
    audit.update(support_indices=support_indices.tolist(), base_scale=base_scale)
    return tilt, coefficients, audit


def load_expression(path, genes=None):
    with np.load(path) as data:
        supplied = data["genes"].astype(str)
        expected = supplied if genes is None else np.asarray(genes).astype(str)
        order = [supplied.tolist().index(gene) for gene in expected]
        return data["x"][:, order].astype(float), expected


def train(wt, ko, condition, output_dir, initial_model_dir=None, fit_state_model=False,
          projection_path=None, support_indices=None, support_size=256, support_seed=1, steps=800):
    """Train one condition, carrying other initialized conditions into the output."""
    condition = CONDITIONS[condition]
    models = {}
    if initial_model_dir is None and not fit_state_model:
        initial_model_dir = task_assets("t3")
    if initial_model_dir is not None:
        initial_model_dir = Path(initial_model_dir)
        with np.load(initial_model_dir / "tilt.npz") as data:
            tilt = dict(data)
        for name in PERTURBED_GENES:
            if (initial_model_dir / f"{name}.npz").exists():
                with np.load(initial_model_dir / f"{name}.npz") as data:
                    models[name] = dict(data)
        x, genes = load_expression(wt, tilt["genes"])
    else:
        x, genes = load_expression(wt)
        tilt = {"genes": genes}
    y, _ = load_expression(ko, genes)
    audit = dict(condition=condition, wt_cells=len(x), ko_cells=len(y), fit_state=fit_state_model)
    if fit_state_model:
        projection = None
        if projection_path is not None:
            with np.load(projection_path) as data:
                if not np.array_equal(data["genes"].astype(str), genes):
                    raise ValueError("Projection gene order must match the training panel")
                projection = {key: data[key] for key in data.files if key.startswith(condition + "_")}
        fitted_tilt, coefficients, audit["state"] = fit_state(
            x, y, genes, condition, support_indices, support_size, support_seed, steps, projection
        )
        tilt.update(fitted_tilt)
    else:
        coefficients = models[condition]["coefficients"]
    models[condition] = calibrate(x, y, tilt, condition, coefficients)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "tilt.npz", **tilt)
    for name, model in models.items():
        np.savez_compressed(output_dir / f"{name}.npz", **model)
    (output_dir / "training.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt", type=Path, required=True, help="WT training NPZ: x and genes")
    parser.add_argument("--ko", type=Path, required=True, help="KO training NPZ: x and genes")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-model-dir", type=Path, help="Initial tilt and condition models; defaults to task assets in frozen mode")
    parser.add_argument("--fit-state", action="store_true", help="Fit PCA8 and mean8 weights before quantile calibration")
    parser.add_argument("--projection", type=Path, help="With --fit-state, reuse this tilt.npz projection instead of refitting PCA")
    parser.add_argument("--support-indices", type=Path, help="NPY indices into ordered WT training rows")
    parser.add_argument("--support-size", type=int, default=256)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    indices = None if args.support_indices is None else np.load(args.support_indices)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        audit = train(args.wt, args.ko, args.condition, args.output_dir, args.initial_model_dir,
                      args.fit_state, args.projection, indices, args.support_size, args.support_seed, args.steps)
    print(json.dumps(dict(output=str(args.output_dir), condition=audit["condition"],
                          wt_cells=audit["wt_cells"], ko_cells=audit["ko_cells"], fit_state=audit["fit_state"])))


if __name__ == "__main__":
    main()
