"""Historical holdout metrics using the frozen training probe."""
import numpy as np
from scipy.spatial.distance import cdist

def vector_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "quantiles": {
            str(q): float(v)
            for q, v in zip((0, .1, .25, .5, .75, .9, 1),
                            np.quantile(values, (0, .1, .25, .5, .75, .9, 1)))
        },
    }

def reference_statistics(matrix, svd, clusters):
    matrix = np.asarray(matrix, dtype=np.float64)
    code = np.asarray(svd.transform(matrix), dtype=np.float64)
    # sklearn's fitted KMeans centers and prediction buffers must share dtype.
    labels = clusters.predict(code.astype(clusters.cluster_centers_.dtype, copy=False))
    return {
        "rows": len(matrix),
        "rate": (matrix > 0).mean(axis=0),
        "mean": matrix.mean(axis=0),
        "variance": matrix.var(axis=0, ddof=0),
        "code": code,
        "covariance": np.cov(code, rowvar=False, ddof=1),
        "clusters": np.bincount(labels, minlength=clusters.n_clusters) / len(labels),
        "detected_gene_count": vector_summary((matrix > 0).sum(axis=1)),
        "linear_total": vector_summary(np.expm1(matrix).sum(axis=1)),
    }

def sample_metrics(prediction, truth, svd, clusters, bandwidth):
    if prediction.ndim != 2 or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid sampled expression")
    pred = reference_statistics(prediction, svd, clusters)
    covariance_denominator = float(np.linalg.norm(truth["covariance"], ord="fro"))
    covariance_denominator = max(covariance_denominator, 1e-12)
    xx = np.exp(-cdist(pred["code"], pred["code"], "sqeuclidean") / bandwidth)
    yy = np.exp(-cdist(truth["code"], truth["code"], "sqeuclidean") / bandwidth)
    xy = np.exp(-cdist(pred["code"], truth["code"], "sqeuclidean") / bandwidth)
    totals = np.expm1(prediction).sum(axis=1)
    return {
        "sampled_detection_mae": float(np.abs(pred["rate"] - truth["rate"]).mean()),
        "sampled_detection_mse": float(np.square(pred["rate"] - truth["rate"]).mean()),
        "pb_log_mean_mae": float(np.abs(pred["mean"] - truth["mean"]).mean()),
        "gene_variance_mae": float(np.abs(pred["variance"] - truth["variance"]).mean()),
        "projected_covariance_fro_relative": float(
            np.linalg.norm(pred["covariance"] - truth["covariance"], ord="fro")
            / covariance_denominator),
        "cluster_tv": float(.5 * np.abs(pred["clusters"] - truth["clusters"]).sum()),
        "mmd_rbf_biased": float(xx.mean() + yy.mean() - 2 * xy.mean()),
        "detected_gene_count": pred["detected_gene_count"],
        "linear_total": pred["linear_total"],
        "total_relative_error_mean": float(np.abs(totals / 10000. - 1).mean()),
        "total_relative_error_max": float(np.abs(totals / 10000. - 1).max()),
        "zero_rows": int(((prediction > 0).sum(axis=1) == 0).sum()),
    }
