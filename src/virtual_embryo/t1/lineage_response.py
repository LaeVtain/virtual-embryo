"""Transfer repeatable within-lineage gene changes and calibrate cardiac atlas capture."""
import numpy as np
from scipy import sparse
from .cardiac_forecast import calibrate, molecule_response


MARKERS={
    "myocardium":("Tnnt2","Tnni1","Actc1","Myl7","Myh6","Myh7","Ttn"),
    "vascular":("Cdh5","Pecam1","Kdr","Esam","Emcn","Klf2","Eng"),
    "epicardium":("Wt1","Tbx18","Aldh1a2","Upk3b","Msln","Upk1b"),
    "crest":("Sox10","Foxd3","Plp1","Tfap2b","Ngfr","Erbb3"),
    "blood":("Hba-a1","Hba-a2","Hbb-bt","Hbb-y","Hbb-bh1","Alas2","Klf1"),
    "surface":("Krt8","Krt18","Krt19","Epcam","Grhl2","Cldn4"),
    "endoderm":("Foxa2","Foxa1","Sox17","Hnf4a","Afp","Ttr","Alb"),
    "mesenchyme":("Pdgfra","Col1a1","Col1a2","Dcn","Prx1","Twist1","Meis1"),
}


def marker_classes(expression, genes):
    """Assign broad cell states from the three strongest measured markers in each panel."""
    scores = []
    for names in MARKERS.values():
        columns = genes.get_indexer(names)
        available = columns[columns >= 0]
        if not len(available):
            scores.append(np.zeros(expression.shape[0]))
            continue
        marker_expression = expression[:, available].toarray()
        scores.append(np.sort(marker_expression, axis=1)[:, -3:].mean(axis=1))
    scores = np.array(scores)
    labels = np.array(list(MARKERS))[np.argmax(scores, axis=0)]
    labels[scores.max(axis=0) == 0] = "mesenchyme"
    return labels


def mean(matrix):
    return np.asarray(matrix.mean(axis=0,dtype=np.float64)).ravel()


def pseudobulk(matrix):
    linear = matrix.astype(np.float64, copy=True)
    linear.data = np.expm1(linear.data)
    return mean(linear)


def make_assigner(cohorts):
    """Source-label marker panels with each lineage represented equally."""
    lineages=sorted(set(np.concatenate([labels for _,labels in cohorts])))
    signatures={key:[] for key in lineages}
    for expression,labels in cohorts:
        means={key:mean(expression[labels==key]) for key in lineages if (labels==key).sum()>=25}
        if len(means)<2:
            continue
        for key,present in means.items():
            others=np.max([mu for k,mu in means.items() if k!=key],axis=0)
            enriched=(present-others)/np.sqrt(0.15+present)
            chosen=np.argsort(enriched)[-32:]
            weights=enriched[chosen].clip(0)
            if weights.sum()>0:
                signatures[key].append((chosen,weights/weights.sum(),present[chosen]))
    def assign(expression):
        scores=[]
        for key in lineages:
            profiles=signatures[key]
            if not profiles:
                scores.append(np.full(expression.shape[0],-1.0))
                continue
            by_study=[]
            for ids,weights,expected in profiles:
                by_study.append(np.asarray(expression[:,ids] @ (weights/np.sqrt(0.2+expected))).ravel())
            scores.append(np.mean(by_study,axis=0))
        return np.array(lineages)[np.argmax(scores,axis=0)]
    return assign


def library_stats(expression,labels,samples):
    out={}
    for kind in np.unique(labels):
        by_replicate=[]
        for lib in np.unique(samples):
            rows=(labels==kind)&(samples==lib)
            if rows.sum()<15:
                continue
            cells=expression[rows]
            counts=np.expm1(cells.data)
            step=np.minimum.reduceat(counts,cells.indptr[:-1])
            by_replicate.append((np.log(pseudobulk(cells)+0.15),np.median(np.log(10000/step)),int(rows.sum())))
        if by_replicate:
            means=np.array([value[0] for value in by_replicate])
            depth=np.median([value[1] for value in by_replicate])
            variance=means.var(axis=0,ddof=1)/len(means) if len(means)>1 else np.full(means.shape[1],0.12)
            out[kind]=(means.mean(0),variance,depth,sum(x[2] for x in by_replicate))
    return out


def estimate(before, after, kind):
    old_mean, old_variance, _, old_size = before[kind]
    new_mean, new_variance, _, new_size = after[kind]
    difference = np.clip(new_mean - old_mean, -2, 2)
    reliability = difference**2 / (
        difference**2 + old_variance + new_variance + 0.12 + 20 / min(old_size, new_size)
    )
    return difference, reliability


def transfer_parameters(early_before,early_after,atlas_before,atlas_after,atlas_later,destination,
                        horizon,early_interval,chip):
    factors={}
    capture={}
    for kind in destination:
        expected=np.log(chip)
        if kind in atlas_before and kind in atlas_after:
            middle,precision=estimate(atlas_before,atlas_after,kind)
            rate=middle/6
            if kind in early_before and kind in early_after:
                slope,early_quality=estimate(early_before,early_after,kind)
                agreement=np.maximum(0,np.sign(slope*middle))
                early_share=0.35*early_quality*agreement
                early_daily=np.clip(slope/early_interval,-0.6,0.6)
                rate=(1-early_share)*rate+early_share*early_daily
            if kind in atlas_later:
                later,quality=estimate(atlas_after,atlas_later,kind)
                precision*=0.65+0.35*quality*(np.sign(later)==np.sign(middle))
            state_weight=0.65*precision
            expected=(1-state_weight)*expected+state_weight*np.clip(rate*horizon,-1.25,1.25)
        factors[kind]=np.exp(expected)
        if kind in atlas_before:
            offset,p=estimate(atlas_before,destination,kind)
            bias=np.exp(0.65*p*np.clip(offset,-1.5,1.5))
            depth_ratio=np.exp(np.clip(destination[kind][2]-atlas_before[kind][2],-1.25,1.25))
            capture[kind]=bias,depth_ratio
    return factors,capture


def by_lineage(donors,labels,factors,default,seed):
    parts=[]
    indices=[]
    for i,kind in enumerate(sorted(set(labels))):
        rows=np.flatnonzero(labels==kind)
        mult=factors.get(kind,default)
        if len(rows)<2:
            parts.append(donors[rows])
        else:
            parts.append(molecule_response(donors[rows],mult,seed+i))
        indices.extend(rows)
    return sparse.vstack(parts,format="csr")[np.argsort(indices)]


def correct_capture(atlas, labels, correction, seed):
    """Thin and replenish molecules to match the early within-lineage measurement offset."""
    rng = np.random.default_rng(seed)
    result = []
    index = []
    for kind in sorted(set(labels)):
        rows=np.flatnonzero(labels==kind)
        if kind not in correction:
            result.append(atlas[rows])
            index.extend(rows)
            continue
        odds, depth = correction[kind]
        all_cells = atlas[rows].astype(np.float64)
        all_cells.data = np.expm1(all_cells.data)
        for start in range(0, len(rows), 32):
            block = all_cells[start:start + 32]
            step = np.minimum.reduceat(block.data, block.indptr[:-1])
            counts = np.rint(block.toarray() / step[:, None]).astype(np.int32)
            multiplier = odds[None, :] * depth * counts.sum(axis=1)[:, None] / (counts @ odds)[:, None]
            kept = rng.binomial(counts, np.minimum(multiplier, 1))
            added = rng.poisson(counts * np.maximum(multiplier - 1, 0))
            molecules = kept + added
            normalized = np.log1p(1e4 * molecules / molecules.sum(1)[:, None])
            result.append(sparse.csr_matrix(normalized, dtype=np.float32))
        index.extend(rows)
    return sparse.vstack(result,format="csr")[np.argsort(index)]


def mix(donors, adjusted, atlas, fraction, seed):
    rows = np.random.default_rng(seed + 239).permutation(donors.shape[0])
    count = round(fraction * len(rows))
    mixed = sparse.vstack([adjusted[rows[count:]], atlas[rows[:count]]], format="csr")
    return calibrate(donors, mixed, adjusted)


