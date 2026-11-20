"""Weight routing and induced-subgraph batching for the published comparisons."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def routed_forward(model, signal, adj, mask, anchor, qa, qanchor, qreadout,
                   feature=None):
    if not bool(mask.any(dim=1).all()):
        raise ValueError('No valid channel: reject before model.')
    h_ts = model.ts_frontend(signal)
    h_anc = model.anc_residual(anchor)
    h_anc = h_anc * qanchor.unsqueeze(-1).float()
    h = h_ts + model.residual_scale * h_anc
    if feature is not None:
        h = h + model.q_feature_adapter(feature.unsqueeze(-1).float()) * mask.unsqueeze(-1).float()
    for block in model.backend.gat_blocks:
        h = block(h, adj, mask, qa)
    g = model.backend.readout(h, mask, quality=qreadout, use_quality=True)
    return model.backend.classifier(g)


def valid_mappings(masks):
    """Permute valid channel indices independently within each participant."""
    maps = []
    for map_number in (1, 2, 3):
        mapping = np.tile(np.arange(masks.shape[1]), (len(masks), 1))
        for subject_index, mask in enumerate(masks):
            valid = np.flatnonzero(mask)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260908, 719, map_number, subject_index])))
            mapping[subject_index, valid] = rng.permutation(valid)
        maps.append(mapping)
    return np.stack(maps)


def condition_q(q, masks, condition, maps):
    q = np.asarray(q, dtype=np.float32)
    if condition == 'Mean_q':
        return (masks * (q.sum(1) / masks.sum(1))[:, None]).astype(np.float32)
    if condition.startswith('Permuted_q_'):
        mapping = maps[int(condition[-1]) - 1]
        return np.take_along_axis(q, mapping, axis=1)
    return q.copy()


def routes(q, mask, condition):
    m = mask.float()
    if condition == 'B_floor': return m, m, m, None
    if condition == 'Readout_only': return m, m, q, None
    if condition == 'No_q_attention': return m, q, q, None
    if condition == 'Q_feature': return m, m, m, q
    if condition == 'Constant_feature': return m, m, m, m
    return q, q, q, None


def forward_ragged(
    model: nn.Module,
    signals: Sequence[torch.Tensor],
    adjacencies: Sequence[torch.Tensor],
    anchors: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not signals or len(signals) != len(adjacencies) or len(signals) != len(anchors):
        raise ValueError("Ragged minibatch lists are empty or misaligned")
    node_counts = [int(signal.shape[1]) for signal in signals]
    if any(signal.ndim != 3 for signal in signals):
        raise ValueError("Each ragged signal must be T x N x 2")
    concatenated_signal = torch.cat(list(signals), dim=1).unsqueeze(0)
    concatenated_anchor = torch.cat(list(anchors), dim=0).unsqueeze(0)
    h_ts = model.ts_frontend(concatenated_signal)
    h_anchor = model.anc_residual(concatenated_anchor)
    h_fused = h_ts + model.residual_scale * h_anchor
    logits: list[torch.Tensor] = []
    start = 0
    for node_count, adjacency in zip(node_counts, adjacencies):
        stop = start + node_count
        subject_h = h_fused[:, start:stop, :]
        mask = torch.ones((1, node_count), dtype=torch.bool, device=subject_h.device)
        logits.append(model.backend(subject_h, adjacency, mask, quality=None))
        start = stop
    if start != int(h_fused.shape[1]):
        raise RuntimeError("Ragged split did not consume every retained node")
    return torch.cat(logits, dim=0)
