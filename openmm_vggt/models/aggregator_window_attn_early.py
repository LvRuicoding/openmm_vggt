from typing import List, Tuple

import torch

from .aggregator import Aggregator, slice_expand_and_flatten


class EarlyFusionAggregator(Aggregator):
    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, in_channels, _, _ = images.shape
        if in_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {in_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * sequence_length, in_channels, images.shape[-2], images.shape[-1])
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        return patch_tokens

    def forward_from_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        image_hw: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], int]:
        image_h, image_w = image_hw
        _, _, channels = patch_tokens.shape

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, sequence_length)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, sequence_length)
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

        _, token_count, _ = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, batch_size, sequence_length, token_count, channels, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, batch_size, sequence_length, token_count, channels, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for idx in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[idx], global_intermediates[idx]], dim=-1)
                output_list.append(concat_inter)

        return output_list, self.patch_start_idx

    def forward(self, images: torch.Tensor, patch_tokens: torch.Tensor | None = None):
        batch_size, sequence_length, _, image_h, image_w = images.shape
        if patch_tokens is None:
            patch_tokens = self.extract_patch_tokens(images)
        return self.forward_from_patch_tokens(
            patch_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            image_hw=(image_h, image_w),
        )
