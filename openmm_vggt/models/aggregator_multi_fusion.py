from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import torch

from .aggregator_window_attn_early import EarlyFusionAggregator


PatchFusionFn = Callable[[torch.Tensor, int], torch.Tensor]


class MultiFusionAggregator(EarlyFusionAggregator):
    """Aggregator that can fuse patch tokens at multiple AA group boundaries.

    Fusion layer ids follow the 24-group alternating-attention convention:
    `0` means before the first frame/global group, and `n > 0` means after
    the n-th complete frame+global group.
    """

    def __init__(
        self,
        *args,
        fusion_layers: Iterable[int] = (0,),
        fusion_methods: Mapping[int | str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fusion_layers = tuple(sorted({int(layer) for layer in fusion_layers}))
        for layer in self.fusion_layers:
            if layer < 0 or layer > self.aa_block_num:
                raise ValueError(
                    f"fusion layer {layer} is out of range [0, {self.aa_block_num}]"
                )

        self.fusion_methods = self._normalize_fusion_methods(fusion_methods)

    def _normalize_fusion_methods(
        self,
        fusion_methods: Mapping[int | str, str] | None,
    ) -> Dict[int, str]:
        if fusion_methods is None:
            return {layer: "window_cross_attn" for layer in self.fusion_layers}

        normalized = {int(layer): str(method) for layer, method in fusion_methods.items()}
        missing_layers = [layer for layer in self.fusion_layers if layer not in normalized]
        if missing_layers:
            raise ValueError(f"Missing fusion methods for layers: {missing_layers}")

        extra_layers = [layer for layer in normalized if layer not in self.fusion_layers]
        if extra_layers:
            raise ValueError(f"Fusion methods specified for disabled layers: {extra_layers}")
        return normalized

    def _apply_patch_fusion_to_tokens(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        token_count: int,
        channels: int,
        layer_id: int,
        patch_fusion_fn: PatchFusionFn,
    ) -> torch.Tensor:
        if tokens.shape != (batch_size * sequence_length, token_count, channels):
            tokens = tokens.view(batch_size, sequence_length, token_count, channels).view(
                batch_size * sequence_length,
                token_count,
                channels,
            )

        patch_tokens = tokens[:, self.patch_start_idx :, :]
        fused_patch_tokens = patch_fusion_fn(patch_tokens, layer_id)
        if fused_patch_tokens.shape != patch_tokens.shape:
            raise ValueError(
                f"Fusion layer {layer_id} returned shape {tuple(fused_patch_tokens.shape)}, "
                f"expected {tuple(patch_tokens.shape)}"
            )

        if self.patch_start_idx == 0:
            return fused_patch_tokens

        return torch.cat([tokens[:, : self.patch_start_idx, :], fused_patch_tokens], dim=1)

    def forward_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        image_hw: Tuple[int, int],
        patch_fusion_fn: PatchFusionFn | None = None,
    ) -> Tuple[List[torch.Tensor], int]:
        image_h, image_w = image_hw
        _, _, channels = patch_tokens.shape

        camera_token = self.camera_token
        register_token = self.register_token
        from .aggregator import slice_expand_and_flatten

        camera_token = slice_expand_and_flatten(camera_token, batch_size, sequence_length)
        register_token = slice_expand_and_flatten(register_token, batch_size, sequence_length)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                batch_size * sequence_length,
                image_h // self.patch_size,
                image_w // self.patch_size,
                device=patch_tokens.device,
            )

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                batch_size * sequence_length,
                self.patch_start_idx,
                2,
                device=patch_tokens.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, token_count, channels = tokens.shape
        if 0 in self.fusion_layers:
            if patch_fusion_fn is None:
                raise ValueError("patch_fusion_fn is required when fusion layer 0 is enabled")
            tokens = self._apply_patch_fusion_to_tokens(
                tokens,
                batch_size,
                sequence_length,
                token_count,
                channels,
                layer_id=0,
                patch_fusion_fn=patch_fusion_fn,
            )

        frame_idx = 0
        global_idx = 0
        output_list = []

        for group_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens,
                        batch_size,
                        sequence_length,
                        token_count,
                        channels,
                        frame_idx,
                        pos=pos,
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens,
                        batch_size,
                        sequence_length,
                        token_count,
                        channels,
                        global_idx,
                        pos=pos,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            layer_id = group_idx + 1
            if layer_id in self.fusion_layers:
                if patch_fusion_fn is None:
                    raise ValueError(f"patch_fusion_fn is required when fusion layer {layer_id} is enabled")
                tokens = self._apply_patch_fusion_to_tokens(
                    tokens,
                    batch_size,
                    sequence_length,
                    token_count,
                    channels,
                    layer_id=layer_id,
                    patch_fusion_fn=patch_fusion_fn,
                )
                global_intermediates[-1] = tokens.view(batch_size, sequence_length, token_count, channels)

            for idx in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[idx], global_intermediates[idx]], dim=-1)
                output_list.append(concat_inter)

        return output_list, self.patch_start_idx

    def forward(
        self,
        images: torch.Tensor,
        patch_tokens: torch.Tensor | None = None,
        patch_fusion_fn: PatchFusionFn | None = None,
    ):
        batch_size, sequence_length, _, image_h, image_w = images.shape
        if patch_tokens is None:
            patch_tokens = self.extract_patch_tokens(images)
        return self.forward_from_patch_tokens(
            patch_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            image_hw=(image_h, image_w),
            patch_fusion_fn=patch_fusion_fn,
        )
