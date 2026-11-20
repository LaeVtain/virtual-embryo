"""Predict T1 expression using a developmental trend and an independent later heart atlas."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .age_prior import gene_multipliers
from .transport import read_stage


def molecule_response(donors, multiplier, seed):
    """Approximate UMI counts, thin decreases and sample gains conditional on local cell state."""
    variance = np.asarray(donors.power(2).mean(axis=0)).ravel() - np.asarray(donors.mean(axis=0)).ravel()**2
    selected = np.argsort(variance)[-1500:]
    n_cells = donors.shape[0]
    components = min(24, len(selected) - 1, n_cells - 1)
    if components < 1:
        raise ValueError("At least two donor cells and two genes are required")
    coordinates = TruncatedSVD(components, random_state=seed).fit_transform(donors[:, selected])
    neighbors = NearestNeighbors(n_neighbors=min(16, n_cells)).fit(coordinates)
    indices = neighbors.kneighbors(coordinates, return_distance=False)
    count = indices.shape[1]
    smoother = sparse.csr_matrix((np.full(indices.size, 1/count),
                                 (np.repeat(np.arange(n_cells), count), indices.ravel())),
                                 shape=(n_cells,n_cells))
    abundance = donors.astype(np.float64, copy=True)
    abundance.eliminate_zeros()
    if np.any(np.diff(abundance.indptr) == 0):
        raise ValueError("Input contains an empty RNA library")
    abundance.data = np.expm1(abundance.data)
    increment = np.maximum(multiplier-1, 0)
    retain = np.minimum(multiplier, 1)
    rng = np.random.default_rng(seed)
    output = []
    for start in range(0,n_cells,64):
        stop = min(start+64,n_cells)
        block = abundance[start:stop]
        step = np.minimum.reduceat(block.data,block.indptr[:-1])
        observed = block.toarray()
        counts = np.rint(observed/step[:,None]).astype(np.int64)
        expected = (observed+(smoother[start:stop] @ abundance).toarray())/(2*step[:,None])
        adjusted = rng.binomial(counts,retain) + rng.poisson(expected*increment)
        totals = adjusted.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("The adjusted molecule counts include an empty library")
        output.append(sparse.csr_matrix(np.log1p(adjusted*1e4/totals[:,None]),dtype=np.float32))
    return sparse.vstack(output, format="csr")


def sample_later(reference, groups, cells, seed):
    replicates = np.unique(groups)
    if len(replicates) != 2:
        raise ValueError("Provide the two independently sequenced heart reference samples")
    rng = np.random.default_rng(seed)
    counts = [cells//2,cells-cells//2]
    selected = [rng.choice(np.flatnonzero(groups==group),size,replace=False)
                for group,size in zip(replicates,counts)]
    return reference[rng.permutation(np.concatenate(selected))]


def calibrate(source, mixture, prior):
    def means(x):
        return np.asarray(x.mean(axis=0,dtype=np.float64)).ravel()

    present = means(source)
    expected = means(prior)
    blended = means(mixture)
    variance = np.maximum(means(source.power(2))-present**2,0)
    trend_variance = (expected-present)**2 + variance/source.shape[0] + 1e-8
    atlas_weight = 0.5*trend_variance/(trend_variance+(blended-expected)**2)
    desired = blended+(1-atlas_weight)*(expected-blended)

    result = sparse.csr_matrix(mixture,dtype=np.float64,copy=True)
    size = np.diff(result.indptr)
    if np.any(size == 0):
        raise ValueError("The mixture contains an empty RNA library")
    mass = np.expm1(result.data)
    for _ in range(6):
        total = np.bincount(result.indices,weights=np.log1p(mass),minlength=result.shape[1])
        derivative = np.bincount(result.indices,weights=mass/(1+mass),minlength=result.shape[1])
        update = np.clip((result.shape[0]*desired-total)/(derivative+3),-0.5,0.5)
        mass *= np.exp(update)[result.indices]
        mass *= np.repeat(10000/np.add.reduceat(mass,result.indptr[:-1]),size)
    result.data = np.log1p(mass)
    return result.astype(np.float32)


def forecast(donors, reference, groups, multiplier, fraction, seed):
    if donors.shape[1] != reference.shape[1] or len(multiplier) != donors.shape[1]:
        raise ValueError("Use the same gene panel for both cohorts and the trend")
    if not 0 < fraction <= 1 or not np.isfinite(multiplier).all() or np.any(multiplier <= 0):
        raise ValueError("Invalid time fraction or gene multiplier")
    adjusted = molecule_response(donors,multiplier,seed)
    later = sample_later(reference,groups,donors.shape[0],seed+31)
    order = np.random.default_rng(seed+239).permutation(donors.shape[0])
    count = round(fraction*len(order))
    mixture = sparse.vstack([adjusted[order[count:]],later[order[:count]]],format="csr")
    return calibrate(donors,mixture,adjusted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True,help="Later heart H5AD with obs['sample']")
    parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--trend",type=Path,required=True,help="Gene-indexed CSV with natural-log log_ratio")
    parser.add_argument("--interval",type=float,required=True,help="Duration represented by log_ratio")
    parser.add_argument("--source-time",type=float,required=True)
    parser.add_argument("--target-time",type=float,required=True)
    parser.add_argument("--reference-time",type=float,default=14.5)
    parser.add_argument("--cells",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.source_time < args.target_time <= args.reference_time:
        parser.error("Provide a new output file and a target between source and reference")
    genes = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    if not genes.is_unique:
        parser.error("Duplicate panel genes")
    source,_ = read_stage(args.source,genes)
    reference,_ = read_stage(args.reference,genes)
    metadata = ad.read_h5ad(args.reference,backed="r")
    try:
        groups = metadata.obs["sample"].to_numpy(copy=True)
    finally:
        metadata.file.close()
    if not 2 <= args.cells <= source.shape[0]:
        parser.error("Requested population size exceeds available source cells or is smaller than two")
    ids = np.random.default_rng(args.seed).choice(source.shape[0],args.cells,replace=False,
                                                  p=np.full(source.shape[0],1/source.shape[0]))
    trend = pd.read_csv(args.trend,index_col=0)["log_ratio"]
    horizon = args.target_time - args.source_time
    factors = gene_multipliers(trend,genes,horizon,args.interval)
    with threadpool_limits(limits=2):
        prediction = forecast(source[ids],reference,groups,factors,horizon/(args.reference_time-args.source_time),
                              args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ad.AnnData(X=prediction,var=pd.DataFrame(index=genes)).write_h5ad(args.output,compression="gzip")
    print(args.output)


if __name__ == "__main__":
    main()
