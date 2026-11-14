"""Memory-bounded CPU inference for the original continuous T2 spatial field.

The learned weights, kernel, global centering, and integration rule are retained.
Dynamic 3-D cKDTree neighbors with distance reordering.
Float-distance ties can select different equivalent-distance neighbors in tree
mode, as can the original topk implementation under row permutations.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def matched_neighbors(n, cap=128):
    return min(cap, int(np.floor(16 * (n - 1) / 127 + .5)) - 1, n - 2)


@torch.inference_mode()
def support_indices(q, k, query_chunk=1024):
    n = len(q)
    tree = cKDTree(q.numpy())
    blocks = []
    for start in range(0, n, query_chunk):
        end = min(n, start + query_chunk)
        _, ix = tree.query(q[start:end].numpy(), k=k + 2, workers=1)
        # Remove the actual row index, including when duplicate coordinates put
        # another zero-distance row before self or omit self from the query.
        ix = torch.from_numpy(np.asarray(ix, dtype=np.int64))
        distances = (q[start:end, None] - q[ix]).square().sum(-1).sqrt()
        distances[ix == torch.arange(start, end)[:, None]] = float("inf")
        order = distances.topk(k + 1, dim=1, largest=False, sorted=True).indices
        blocks.append(torch.gather(ix, 1, order))
    return torch.cat(blocks)


@torch.inference_mode()
def landmark_support(q, anchors, k, query_chunk=1024):
    """Dynamic neighbors within a source-fixed random landmark subset."""
    tree = cKDTree(q[anchors].numpy())
    blocks = []
    for start in range(0, len(q), query_chunk):
        end = min(len(q), start + query_chunk)
        _, local_ix = tree.query(q[start:end].numpy(), k=k + 2, workers=1)
        ix = anchors[torch.from_numpy(np.asarray(local_ix, dtype=np.int64))]
        distances = (q[start:end, None] - q[ix]).square().sum(-1).sqrt()
        distances[ix == torch.arange(start, end)[:, None]] = float("inf")
        order = distances.topk(k + 1, dim=1, largest=False, sorted=True).indices
        blocks.append(torch.gather(ix, 1, order))
    return torch.cat(blocks)



def field_values(net, q, embedding, radius2, support, start, end, k, features=None):
    # The dense training path retains the original autograd operation order.
    dense = features is not None
    qq = q if dense else q[start:end]
    all_relative = qq[:, None, :] - q[support]
    all_squared = all_relative.square().sum(dim=-1, keepdim=True)
    ix = support[:, :k]
    relative, squared = all_relative[:, :k], all_squared[:, :k]
    support_squared = all_squared[:, -1:, :]
    positive = support_squared > 0
    denominator = torch.where(positive, support_squared, torch.ones_like(support_squared))
    phi = torch.clamp(1 - squared / denominator, min=0).square()
    phi = torch.where(positive, phi, torch.zeros_like(phi))
    weights = phi / (phi.sum(dim=1, keepdim=True) + 1e-6)
    neighbor_q = q[ix]
    if dense:
        radius2 = q.square().sum(dim=-1, keepdim=True)
    rr = radius2 if dense else radius2[start:end]
    neighbor_radius2 = radius2[ix]
    dot = (qq[:, None, :] * neighbor_q).sum(dim=-1, keepdim=True)
    if dense:
        embedding = net.feature_encoder(features)
    center = embedding * 1.0 if dense else embedding[start:end]
    neighbor = embedding[ix] * 1.0 if dense else embedding[ix]
    mean_distance2 = squared.mean(dim=1)
    radial_input = torch.cat((center, torch.log1p(rr), torch.log1p(mean_distance2)), dim=1)
    edge_input = torch.cat((center[:, None, :].expand(-1, k, -1), neighbor,
                           torch.log1p(squared), torch.log1p(rr)[:, None, :].expand(-1, k, -1),
                           torch.log1p(neighbor_radius2), dot / (1 + dot.abs())), dim=-1)
    edge_vector = relative / (1 + torch.sqrt(squared + 1e-12))
    local = (weights * net.edge(edge_input) * edge_vector).sum(dim=1)
    radial_vector = qq / (1 + torch.sqrt(rr + 1e-12))
    return net.radial(radial_input) * radial_vector + local


def dense_field(net, positions, features):
    """Differentiable training field; dense neighbor selection matches training."""
    if len(positions)<=net.k+1:
        raise ValueError("Continuous support needs at least k+2 points")
    q=positions-positions.mean(dim=0,keepdim=True)
    with torch.no_grad():
        distances=torch.cdist(q,q,compute_mode="donot_use_mm_for_euclid_dist")
        distances.fill_diagonal_(float("inf"))
        support=distances.topk(net.k+1,dim=1,largest=False,sorted=True).indices
    return field_values(net,q,None,None,support,0,len(q),net.k,features=features)


def spatial_features(x,start,end,tmin,span):
    return torch.cat((x,x.new_full((len(x),1),(start-tmin)/span),
                      x.new_full((len(x),1),(end-start)/span)),1)


def training_rollout(net,coordinates,features,duration,steps=8):
    value=coordinates-coordinates.mean(0)
    for _ in range(steps):
        velocity=net(value,features)
        value=value+(velocity-velocity.mean(0))*duration/steps
    return value

class SpatialField:
    """A fixed feature embedding shared by every step of one source rollout."""
    def __init__(self, net, features, *, chunk_size=1024):
        self.net = net
        self.chunk_size = chunk_size
        with torch.inference_mode():
            self.embedding = net.feature_encoder(features)

    @torch.inference_mode()
    def __call__(self, positions, k=15, anchors=None):
        net, embedding = self.net, self.embedding
        q = positions - positions.mean(dim=0, keepdim=True)
        support = (support_indices(q, k, self.chunk_size) if anchors is None
                   else landmark_support(q, anchors, k, self.chunk_size))
        radius2 = q.square().sum(-1, keepdim=True)
        output = torch.empty_like(q)
        for start in range(0, len(q), self.chunk_size):
            end = min(len(q), start + self.chunk_size)
            ss = support[start:end]
            output[start:end] = field_values(net,q,embedding,radius2,ss,start,end,k)
        return output


@torch.inference_mode()
def rollout(net, coordinates, features, duration, *, steps=32, k=15, chunk_size=1024):
    field = SpatialField(net, features, chunk_size=chunk_size)
    result = coordinates - coordinates.mean(0)
    for _ in range(steps):
        velocity = field(result, k)
        result = result + (velocity - velocity.mean(0)) * duration / steps
    return result


@torch.inference_mode()
def landmark_rollout(net, coordinates, features, duration, *, anchor_count=128,
                     anchor_k=15, seed=20261111, steps=32, chunk_size=1024):
    result = coordinates - coordinates.mean(0)
    if len(result) == 128:
        return rollout(net, coordinates, features, duration, steps=steps, chunk_size=chunk_size)
    field = SpatialField(net, features, chunk_size=chunk_size)
    count = min(anchor_count, len(result))
    anchors = torch.from_numpy(np.sort(np.random.default_rng(seed).choice(len(result),count,replace=False)))
    k = min(anchor_k, matched_neighbors(count, cap=anchor_k))
    for _ in range(steps):
        velocity = field(result,k,anchors)
        result = result + (velocity-velocity.mean(0))*duration/steps
    return result
