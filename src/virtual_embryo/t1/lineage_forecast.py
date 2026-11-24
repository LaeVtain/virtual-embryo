"""Forecast a cardiac RNA population using replicate-supported lineage changes."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .age_prior import gene_multipliers
from .cardiac_forecast import sample_later
from .lineage_response import (
    by_lineage, correct_capture, library_stats, make_assigner,
    marker_classes, mix, transfer_parameters,
)
from .transport import read_stage


def sample_libraries(libraries, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for library in np.unique(libraries):
        rows = np.flatnonzero(libraries == library)
        selected.append(rng.choice(rows, min(len(rows), limit), replace=False))
    return np.concatenate(selected)


def read_measurement(path, panel):
    expression, _ = read_stage(path, panel)
    data = ad.read_h5ad(path, backed="r")
    try:
        if "sample" in data.obs:
            libraries = data.obs["sample"].astype(str).to_numpy()
        else:
            libraries = np.array([barcode.rsplit("_", 1)[-1] for barcode in data.obs_names])
    finally:
        data.file.close()
    return expression, libraries


def forecast(donors, panel, bundle, heart_atlas, heart_samples, calibration, calibration_samples,
             multipliers, source_time, target_time, seed, reference_assay="whole-cell"):
    if not 8.25 <= source_time < target_time < 14.5:
        raise ValueError("Use a source from E8.25 onward and a target before the E14.5 heart reference.")
    if not bundle.var_names.equals(panel):
        raise ValueError("The reference bundle must have the requested gene panel and column order.")
    stages = bundle.obs["cohort"].astype(str).to_numpy()
    groups = bundle.obs["sample"].astype(str).to_numpy()

    def cohort(name):
        ids = np.flatnonzero(stages == name)
        if not len(ids):
            raise ValueError(f"Missing reference cohort {name}")
        expression = bundle.X[ids].tocsr()
        return expression, marker_classes(expression, panel), groups[ids]

    earlier, recent = ("7.75", "8.25") if source_time < 9.25 else ("8.25", "9.25")
    early_old = cohort("wt_" + earlier)
    early_new = cohort("wt_" + recent)
    source_lineages = marker_classes(donors, panel)
    assign = make_assigner([(early_new[0], early_new[1]), (donors, source_lineages)])

    def atlas_stats(expression, samples):
        return library_stats(expression, assign(expression), samples)

    young_atlas, _, young_samples = cohort("10x_8.5")
    late_atlas, _, late_samples = cohort("10x_16.5")
    middle = sample_libraries(heart_samples, 1500, 716)
    calibration_rows = sample_libraries(calibration_samples, 1000, 39)
    calibration = calibration[calibration_rows]
    calibration_stats = library_stats(calibration, marker_classes(calibration, panel),
                                      calibration_samples[calibration_rows])
    factors, adjustments = transfer_parameters(
        library_stats(*early_old), library_stats(*early_new),
        atlas_stats(young_atlas, young_samples), atlas_stats(heart_atlas[middle], heart_samples[middle]),
        atlas_stats(late_atlas, late_samples), calibration_stats,
        target_time - source_time, float(recent) - float(earlier), multipliers,
    )
    if reference_assay == "nucleus":
        if len(np.unique(heart_samples)) != 1:
            raise ValueError("The nuclear reference uses the single E14.5 wild-type GSM8394736 library.")
        rng = np.random.default_rng(seed + 31)
        ids = rng.choice(heart_atlas.shape[0], donors.shape[0], replace=False)
        sampled = heart_atlas[ids].tocsr()
    elif reference_assay == "whole-cell":
        sampled = sample_later(heart_atlas, heart_samples, donors.shape[0], seed + 31)
    else:
        raise ValueError(f"Unknown reference assay {reference_assay}")
    corrected = correct_capture(sampled, assign(sampled), adjustments, seed + 305)
    projected = by_lineage(donors, source_lineages, factors, multipliers, seed)
    fraction = (target_time - source_time) / (14.5 - source_time)
    return mix(donors, projected, corrected, fraction, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="Public E8.5 RNA, with library identifiers")
    parser.add_argument("--reference", type=Path, required=True, help="Wild-type E14.5 heart H5AD")
    parser.add_argument("--reference-assay", choices=["whole-cell", "nucleus"], default="whole-cell",
                        help="Use the two-library whole-cell atlas or independently assayed GSM8394736 nuclei")
    parser.add_argument("--nuclear-reference", type=Path,
                        help="Use this second E14.5 nuclear atlas for expression marginals on whole-cell ranks")
    parser.add_argument("--nuclear-trajectory", action="store_true",
                        help="Calibrate gene means across the source and late nuclei at matched detection complexity")
    parser.add_argument("--nuclear-population", choices=("rank", "odds"),
                        help="Refine the nuclear trajectory using the full, measured source population")
    parser.add_argument("--empirical-fraction", type=float, default=0.,
                        help="Fraction of nuclear forecast cells based on intact depth-matched measurements")
    parser.add_argument("--lineage-reference", type=Path, required=True, help="Prepared external lineage H5AD")
    parser.add_argument("--chamber-reference", type=Path, help="Optional WT cardiac chamber time-course NPZ")
    parser.add_argument("--state-dispersion", action="store_true",
                        help="Adjust cell-state covariance at fixed detection and gene means")
    parser.add_argument("--cross-gene-dispersion", action="store_true",
                        help="Apply state dispersion and adjust residual covariance using disjoint gene sets")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--trend", type=Path, required=True)
    parser.add_argument("--interval", type=float, required=True, help="Days spanned by the trend")
    parser.add_argument("--source-time", type=float, required=True)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    if args.nuclear_reference and args.reference_assay != "whole-cell":
        parser.error("--nuclear-reference requires a whole-cell atlas as the primary --reference")
    if args.nuclear_trajectory and not args.nuclear_reference:
        parser.error("--nuclear-trajectory requires --nuclear-reference")
    if args.nuclear_population and not args.nuclear_trajectory:
        parser.error("--nuclear-population requires --nuclear-trajectory")
    if not 0 <= args.empirical_fraction <= 1 or (args.empirical_fraction and not args.nuclear_population):
        parser.error("--empirical-fraction must be in [0, 1] and requires --nuclear-population when positive")
    genes = pd.Index(args.panel.read_text(encoding="utf-8-sig").splitlines())
    source, _ = read_stage(args.source, genes)
    if not 2 <= args.cells <= source.shape[0]:
        parser.error("Specify between two and the source population size for --cells")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(source.shape[0], args.cells, replace=False,
                          p=np.full(source.shape[0], 1 / source.shape[0]))
    donors = source[selected]
    del source
    calibration, earlier_libraries = read_measurement(args.calibration, genes)
    bundle = ad.read_h5ad(args.lineage_reference)
    trend = pd.read_csv(args.trend, index_col=0)["log_ratio"]
    multipliers = gene_multipliers(trend, genes, args.target_time - args.source_time, args.interval)

    def predict(measurement, assay):
        reference, libraries = measurement
        output = forecast(donors, genes, bundle, reference, libraries, calibration,
                          earlier_libraries, multipliers, args.source_time, args.target_time, args.seed,
                          assay)
        if args.chamber_reference:
            from .chamber_projection import project_chambers
            output = project_chambers(donors, output, genes, args.chamber_reference,
                                      args.source_time, args.target_time)
        if args.state_dispersion or args.cross_gene_dispersion:
            from .state_dispersion import adjust_state_dispersion
            output = adjust_state_dispersion(output, genes)
        if args.cross_gene_dispersion:
            from .cross_gene_dispersion import adjust_cross_gene_dispersion
            output = adjust_cross_gene_dispersion(output, genes)
        return output

    with threadpool_limits(limits=2):
        output = predict(read_measurement(args.reference, genes), args.reference_assay)
        if args.nuclear_reference:
            from .reference_marginals import exchange_marginals, project_nuclear_means
            measurement = read_measurement(args.nuclear_reference, genes)
            nuclear = predict(measurement, "nucleus")
            del calibration, bundle
            output = exchange_marginals(output, nuclear)
            del nuclear
            if args.nuclear_trajectory:
                output = project_nuclear_means(output, donors, measurement[0], args.source_time, args.target_time)
            if args.nuclear_population:
                from .nuclear_population import project_population
                del donors
                population, _ = read_stage(args.source, genes)
                output = project_population(output, population, measurement[0], args.source_time,
                                            args.target_time, method=args.nuclear_population)
                if args.empirical_fraction:
                    from .nuclear_population import restore_empirical_cells
                    fraction = (args.target_time - args.source_time) / (14.5 - args.source_time)
                    output = restore_empirical_cells(output, population, measurement[0], fraction, args.seed,
                                                     args.empirical_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(output, var=pd.DataFrame(index=genes)).write_h5ad(args.output, compression="gzip")
    print(args.output)


if __name__ == "__main__":
    main()
