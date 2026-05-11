# Multi-Fusion Aggregator Design

## Goal

Add a new multi-fusion occupancy model path that can inject LiDAR voxel features into VGGT patch tokens multiple times:

- before the aggregator starts, after patch embedding;
- after selected `frame + global` alternating-attention groups;
- with a per-layer configurable fusion method.

The existing `EarlyFusionAggregator` and existing model configs must keep their current behavior.

## Layer Numbering

The aggregator has `depth=24` alternating-attention groups by default. Each group runs:

```text
frame_i -> global_i
```

Layer ids use group numbering:

- `0`: after patch embedding and before `frame_0`;
- `12`: after `frame_11 -> global_11`;
- `24`: after `frame_23 -> global_23`.

This is not the 48-block attention count. It is the 24-group `frame + global` count.

## Fusion Points

For the initial experiment, use:

```python
fusion_layers = (0, 12, 24)
fusion_methods = {
    0: "window_cross_attn",
    12: "window_cross_attn",
    24: "window_cross_attn",
}
fusion_share_weights = False
```

Fusion runs only on patch tokens. Camera/register tokens are preserved.

The layer-12 and layer-24 fusion steps run after the complete `frame + global` group. The fused patch tokens become the input to the next group. For layer 24, the fused tokens must also be reflected in the final output feature sent to the occupancy head.

## Feature Dimensions

Fusion happens inside the aggregator, where patch tokens are `embed_dim=1024`. The occupancy head receives concatenated frame/global features later, so it still sees `2 * embed_dim = 2048` features.

## Fusion Inputs

Voxel encoding, projection, visibility selection, and patch assignment are independent of aggregator layer tokens, so compute these once per forward pass and reuse them for all fusion layers.

## Parameter Sharing

Do not share fusion parameters across layers. Each configured fusion layer owns its own fusion module. The voxel encoder can stay shared.

## Supported Methods

Initial implementation supports:

- `"window_cross_attn"`: current shift-window patch-to-voxel cross-attention method.

The implementation should be structured so new methods such as `"add"`, `"gate"`, `"concat_mlp"`, or `"serializer2d"` can be added later.

## Checkpoint Strategy

The new model introduces parameters absent from old checkpoints. Continue from an old checkpoint with non-strict model loading, and rebuild optimizer state. New fusion modules will be randomly initialized.

## Implementation Plan

1. Add `MultiFusionAggregator`, derived from `EarlyFusionAggregator`.
2. Add a model class that uses `MultiFusionAggregator` and the existing occupancy head path.
3. Add config fields:
   - `fusion_layers`
   - `fusion_methods`
   - `fusion_share_weights`
4. Add a new occupancy config instead of modifying the old one.
5. Run syntax/import checks.
