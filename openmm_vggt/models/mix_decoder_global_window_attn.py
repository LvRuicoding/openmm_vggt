from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

from .mix_decoder_global import mix_decoder_global

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


VALID_ATTN_BACKENDS = {"auto", "flash", "sdpa", "math"}


class ControlledAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 16, attn_backend: str = "auto"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if attn_backend not in VALID_ATTN_BACKENDS:
            raise ValueError(f"attn_backend must be one of {sorted(VALID_ATTN_BACKENDS)}, but got {attn_backend}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_backend = attn_backend

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _build_sdpa_mask(
        self,
        key_padding_mask: torch.Tensor | None,
        q: torch.Tensor,
    ) -> torch.Tensor | None:
        if key_padding_mask is None:
            return None
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(torch.bool)
        mask = torch.zeros(
            (q.shape[0], 1, 1, key_padding_mask.shape[1]),
            dtype=q.dtype,
            device=q.device,
        )
        return mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

    def _flash_is_usable(self, q: torch.Tensor, key_padding_mask: torch.Tensor | None) -> bool:
        return (
            FLASH_ATTN_AVAILABLE
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and key_padding_mask is None
        )

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_flash = q.transpose(1, 2)
        k_flash = k.transpose(1, 2)
        v_flash = v.transpose(1, 2)
        output = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0)
        return output.reshape(q.shape[0], q.shape[2], self.embed_dim)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = self._build_sdpa_mask(key_padding_mask, q)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        return output.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.embed_dim)

    def _math_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        output = torch.matmul(attn, v)
        return output.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))

        if key_padding_mask is not None and not torch.any(key_padding_mask):
            key_padding_mask = None

        if self.attn_backend == "flash":
            if key_padding_mask is not None:
                raise ValueError("attn_backend='flash' does not support key_padding_mask in this implementation")
            if not self._flash_is_usable(q, None):
                raise RuntimeError(
                    "attn_backend='flash' requires flash_attn, CUDA, and fp16/bf16 inputs; "
                    "otherwise use 'sdpa', 'math', or 'auto'"
                )
            output = self._flash_attention(q, k, v)
        elif self.attn_backend == "sdpa":
            output = self._sdpa_attention(q, k, v, key_padding_mask=key_padding_mask)
        elif self.attn_backend == "math":
            output = self._math_attention(q, k, v, key_padding_mask=key_padding_mask)
        else:
            if self._flash_is_usable(q, key_padding_mask):
                output = self._flash_attention(q, k, v)
            else:
                output = self._sdpa_attention(q, k, v, key_padding_mask=key_padding_mask)

        return self.out_proj(output)


class LocalPatchVoxelAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ControlledAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_backend=attn_backend,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_tokens = self.norm1(tokens)
        attn_out = self.attn(
            attn_tokens,
            attn_tokens,
            attn_tokens,
            key_padding_mask=key_padding_mask,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        if key_padding_mask is not None:
            tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return tokens


class LocalPatchVoxelCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = ControlledAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_backend=attn_backend,
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        voxel_tokens: torch.Tensor,
        query_padding_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if voxel_tokens.shape[1] == 0:
            output = patch_tokens
        else:
            norm_patch_tokens = self.query_norm(patch_tokens)
            norm_voxel_tokens = self.kv_norm(voxel_tokens)
            attn_out = self.attn(
                norm_patch_tokens,
                norm_voxel_tokens,
                norm_voxel_tokens,
                key_padding_mask=key_padding_mask,
            )
            output = patch_tokens + attn_out
            output = output + self.mlp(self.out_norm(output))

        if query_padding_mask is not None:
            output = output.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        return output


class SlidingWindowPatchVoxelFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        self.window_h = int(window_size[0])
        self.window_w = int(window_size[1])
        self.stride_h = int(stride[0])
        self.stride_w = int(stride[1])
        if self.window_h <= 0 or self.window_w <= 0:
            raise ValueError(f"window_size must be positive, but got {window_size}")
        if self.stride_h <= 0 or self.stride_w <= 0:
            raise ValueError(f"stride must be positive, but got {stride}")
        if self.stride_h >= self.window_h or self.stride_w >= self.window_w:
            raise ValueError(
                f"stride must be smaller than window_size, but got stride={stride}, "
                f"window_size={window_size}"
            )
        self.block = LocalPatchVoxelAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_backend=attn_backend,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        voxel_tokens: torch.Tensor,
        voxel_patch_y: torch.Tensor,
        voxel_patch_x: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        channels = patch_tokens.shape[-1]
        patch_grid = patch_tokens.reshape(patch_h, patch_w, channels)
        accum = torch.zeros_like(patch_grid)
        counts = patch_tokens.new_zeros((patch_h, patch_w, 1))
        window_area = self.window_h * self.window_w
        device = patch_tokens.device

        for y0 in range(0, patch_h, self.stride_h):
            for x0 in range(0, patch_w, self.stride_w):
                y1 = min(y0 + self.window_h, patch_h)
                x1 = min(x0 + self.window_w, patch_w)
                valid_h = y1 - y0
                valid_w = x1 - x0

                valid_patch_mask = torch.zeros(
                    (self.window_h, self.window_w),
                    dtype=torch.bool,
                    device=device,
                )
                valid_patch_mask[:valid_h, :valid_w] = True
                flat_valid_patch_mask = valid_patch_mask.reshape(-1)

                window_patch_tokens = patch_tokens.new_zeros((window_area, channels))
                window_patch_tokens[flat_valid_patch_mask] = patch_grid[y0:y1, x0:x1].reshape(-1, channels)

                in_window = (
                    (voxel_patch_y >= y0)
                    & (voxel_patch_y < y0 + self.window_h)
                    & (voxel_patch_x >= x0)
                    & (voxel_patch_x < x0 + self.window_w)
                )
                window_voxel_tokens = voxel_tokens[in_window]
                window_tokens = torch.cat([window_patch_tokens, window_voxel_tokens], dim=0)

                key_padding_mask = torch.zeros(
                    (1, window_tokens.shape[0]),
                    dtype=torch.bool,
                    device=device,
                )
                key_padding_mask[:, :window_area] = ~flat_valid_patch_mask.unsqueeze(0)
                window_tokens = self.block(
                    window_tokens.unsqueeze(0),
                    key_padding_mask=key_padding_mask,
                ).squeeze(0)

                updated_patches = window_tokens[:window_area][flat_valid_patch_mask]
                accum[y0:y1, x0:x1] += updated_patches.reshape(valid_h, valid_w, channels)
                counts[y0:y1, x0:x1] += 1.0

        fused_grid = accum / counts.clamp_min(1.0)
        return fused_grid.reshape(patch_h * patch_w, channels)


class SlidingWindowPatchVoxelCrossFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        self.window_h = int(window_size[0])
        self.window_w = int(window_size[1])
        self.stride_h = int(stride[0])
        self.stride_w = int(stride[1])
        if self.window_h <= 0 or self.window_w <= 0:
            raise ValueError(f"window_size must be positive, but got {window_size}")
        if self.stride_h <= 0 or self.stride_w <= 0:
            raise ValueError(f"stride must be positive, but got {stride}")
        if self.stride_h >= self.window_h or self.stride_w >= self.window_w:
            raise ValueError(
                f"stride must be smaller than window_size, but got stride={stride}, "
                f"window_size={window_size}"
            )
        self.block = LocalPatchVoxelCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_backend=attn_backend,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        voxel_tokens: torch.Tensor,
        voxel_patch_y: torch.Tensor,
        voxel_patch_x: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        channels = patch_tokens.shape[-1]
        patch_grid = patch_tokens.reshape(patch_h, patch_w, channels)
        accum = torch.zeros_like(patch_grid)
        counts = patch_tokens.new_zeros((patch_h, patch_w, 1))
        window_area = self.window_h * self.window_w
        device = patch_tokens.device

        for y0 in range(0, patch_h, self.stride_h):
            for x0 in range(0, patch_w, self.stride_w):
                y1 = min(y0 + self.window_h, patch_h)
                x1 = min(x0 + self.window_w, patch_w)
                valid_h = y1 - y0
                valid_w = x1 - x0

                valid_patch_mask = torch.zeros(
                    (self.window_h, self.window_w),
                    dtype=torch.bool,
                    device=device,
                )
                valid_patch_mask[:valid_h, :valid_w] = True
                flat_valid_patch_mask = valid_patch_mask.reshape(-1)

                window_patch_tokens = patch_tokens.new_zeros((window_area, channels))
                window_patch_tokens[flat_valid_patch_mask] = patch_grid[y0:y1, x0:x1].reshape(-1, channels)

                in_window = (
                    (voxel_patch_y >= y0)
                    & (voxel_patch_y < y0 + self.window_h)
                    & (voxel_patch_x >= x0)
                    & (voxel_patch_x < x0 + self.window_w)
                )
                window_voxel_tokens = voxel_tokens[in_window]
                if window_voxel_tokens.shape[0] == 0:
                    updated_patches = window_patch_tokens[flat_valid_patch_mask]
                else:
                    window_patch_tokens = self.block(
                        window_patch_tokens.unsqueeze(0),
                        window_voxel_tokens.unsqueeze(0),
                        query_padding_mask=(~flat_valid_patch_mask).unsqueeze(0),
                    ).squeeze(0)
                    updated_patches = window_patch_tokens[flat_valid_patch_mask]

                accum[y0:y1, x0:x1] += updated_patches.reshape(valid_h, valid_w, channels)
                counts[y0:y1, x0:x1] += 1.0

        fused_grid = accum / counts.clamp_min(1.0)
        return fused_grid.reshape(patch_h * patch_w, channels)


class ShiftWindowPatchVoxelCrossFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int] | None = None,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        self.window_h = int(window_size[0])
        self.window_w = int(window_size[1])
        if self.window_h <= 0 or self.window_w <= 0:
            raise ValueError(f"window_size must be positive, but got {window_size}")

        if shift_size is None:
            shift_size = (self.window_h // 2, self.window_w // 2)
        self.shift_h = int(shift_size[0])
        self.shift_w = int(shift_size[1])
        if self.shift_h < 0 or self.shift_w < 0:
            raise ValueError(f"shift_size must be non-negative, but got {shift_size}")
        if self.shift_h >= self.window_h or self.shift_w >= self.window_w:
            raise ValueError(
                f"shift_size must be smaller than window_size, but got shift_size={shift_size}, "
                f"window_size={window_size}"
            )

        self.regular_block = LocalPatchVoxelCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_backend=attn_backend,
        )
        self.shifted_block = LocalPatchVoxelCrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_backend=attn_backend,
        )

    @staticmethod
    def _pad_to_window(length: int, window: int) -> int:
        return ((length + window - 1) // window) * window

    def _run_stage(
        self,
        patch_grid: torch.Tensor,
        voxel_tokens: torch.Tensor,
        voxel_patch_y: torch.Tensor,
        voxel_patch_x: torch.Tensor,
        shift_h: int,
        shift_w: int,
        block: LocalPatchVoxelCrossAttentionBlock,
    ) -> torch.Tensor:
        patch_h, patch_w, channels = patch_grid.shape
        padded_h = self._pad_to_window(patch_h, self.window_h)
        padded_w = self._pad_to_window(patch_w, self.window_w)
        device = patch_grid.device
        window_area = self.window_h * self.window_w

        output_grid = patch_grid.clone()

        for y0 in range(-shift_h, padded_h, self.window_h):
            y1 = y0 + self.window_h
            valid_y0 = max(y0, 0)
            valid_y1 = min(y1, patch_h)
            valid_h = max(valid_y1 - valid_y0, 0)

            for x0 in range(-shift_w, padded_w, self.window_w):
                x1 = x0 + self.window_w
                valid_x0 = max(x0, 0)
                valid_x1 = min(x1, patch_w)
                valid_w = max(valid_x1 - valid_x0, 0)

                valid_patch_mask = torch.zeros((self.window_h, self.window_w), dtype=torch.bool, device=device)
                window_patch_tokens = patch_grid.new_zeros((window_area, channels))

                if valid_h > 0 and valid_w > 0:
                    local_y0 = valid_y0 - y0
                    local_x0 = valid_x0 - x0
                    valid_patch_mask[local_y0 : local_y0 + valid_h, local_x0 : local_x0 + valid_w] = True
                    flat_valid_patch_mask = valid_patch_mask.reshape(-1)
                    window_patch_tokens[flat_valid_patch_mask] = patch_grid[valid_y0:valid_y1, valid_x0:valid_x1].reshape(
                        -1, channels
                    )
                else:
                    flat_valid_patch_mask = valid_patch_mask.reshape(-1)

                in_window = (
                    (voxel_patch_y >= y0)
                    & (voxel_patch_y < y1)
                    & (voxel_patch_x >= x0)
                    & (voxel_patch_x < x1)
                )
                window_voxel_tokens = voxel_tokens[in_window]
                if window_voxel_tokens.shape[0] > 0:
                    window_patch_tokens = block(
                        window_patch_tokens.unsqueeze(0),
                        window_voxel_tokens.unsqueeze(0),
                        query_padding_mask=(~flat_valid_patch_mask).unsqueeze(0),
                    ).squeeze(0)

                if valid_h > 0 and valid_w > 0:
                    output_grid[valid_y0:valid_y1, valid_x0:valid_x1] = window_patch_tokens[flat_valid_patch_mask].reshape(
                        valid_h, valid_w, channels
                    )

        return output_grid

    def forward(
        self,
        patch_tokens: torch.Tensor,
        voxel_tokens: torch.Tensor,
        voxel_patch_y: torch.Tensor,
        voxel_patch_x: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        patch_grid = patch_tokens.reshape(patch_h, patch_w, patch_tokens.shape[-1])
        patch_grid = self._run_stage(
            patch_grid,
            voxel_tokens,
            voxel_patch_y,
            voxel_patch_x,
            shift_h=0,
            shift_w=0,
            block=self.regular_block,
        )
        if self.shift_h > 0 or self.shift_w > 0:
            patch_grid = self._run_stage(
                patch_grid,
                voxel_tokens,
                voxel_patch_y,
                voxel_patch_x,
                shift_h=self.shift_h,
                shift_w=self.shift_w,
                block=self.shifted_block,
            )
        return patch_grid.reshape(patch_h * patch_w, patch_tokens.shape[-1])


@MODELS.register_module()
class mix_decoder_global_window_attn(mix_decoder_global):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        cam_num=2,
        voxel_size=(0.4, 0.4, 0.8),
        point_cloud_range=(0.0, -40.0, -3.0, 80.0, 40.0, 3.0),
        voxel_encoder_filters=(128, 128),
        serializer_grid_size_2d=14.0,
        fusion_window_sizes: Sequence[Tuple[int, int]] = ((4, 6), (6, 8), (8, 10), (10, 12)),
        fusion_window_strides: Sequence[Tuple[int, int]] = ((2, 3), (3, 4), (4, 5), (5, 6)),
        fusion_num_heads: int = 16,
        fusion_mlp_ratio: float = 4.0,
        fusion_attn_backend: str = "sdpa",
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            cam_num=cam_num,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            voxel_encoder_filters=voxel_encoder_filters,
            serializer_grid_size_2d=serializer_grid_size_2d,
        )
        if len(fusion_window_sizes) != len(self.selected_list):
            raise ValueError(
                f"fusion_window_sizes must match selected_list length {len(self.selected_list)}, "
                f"but got {len(fusion_window_sizes)}"
            )
        if len(fusion_window_strides) != len(self.selected_list):
            raise ValueError(
                f"fusion_window_strides must match selected_list length {len(self.selected_list)}, "
                f"but got {len(fusion_window_strides)}"
            )

        self.window_fusions = nn.ModuleList(
            [
                SlidingWindowPatchVoxelFusion(
                    embed_dim=embed_dim * 2,
                    window_size=tuple(window_size),
                    stride=tuple(window_stride),
                    num_heads=fusion_num_heads,
                    mlp_ratio=fusion_mlp_ratio,
                    attn_backend=fusion_attn_backend,
                )
                for window_size, window_stride in zip(fusion_window_sizes, fusion_window_strides)
            ]
        )
        self.patch_fusions = nn.ModuleList()

    def _prepare_patch_voxel_features(
        self,
        others: dict,
        batch_size: int,
        frame_count: int,
        image_hw,
    ) -> Dict[str, torch.Tensor | int]:
        points = others["points"].reshape(
            batch_size * frame_count,
            others["points"].shape[2],
            others["points"].shape[3],
        )
        point_mask = others["point_mask"].reshape(batch_size * frame_count, others["point_mask"].shape[2])
        intrinsics = others["intrinsics"].reshape(batch_size, frame_count, self.cam_num, 3, 3)
        intrinsics = intrinsics.reshape(batch_size * frame_count, self.cam_num, 3, 3)
        camera_to_world = others["camera_to_world"].reshape(batch_size, frame_count, self.cam_num, 4, 4)
        camera_to_world = camera_to_world.reshape(batch_size * frame_count, self.cam_num, 4, 4)

        lidar_to_world = None
        if "lidar_to_world" in others:
            lidar_to_world = others["lidar_to_world"].reshape(batch_size * frame_count, 4, 4)

        voxel_features, voxel_coords = self._encode_voxels(points, point_mask, lidar_to_world=lidar_to_world)
        voxel_tokens = self.voxel_feature_proj(voxel_features)
        patch_h = image_hw[0] // self.patch_size
        patch_w = image_hw[1] // self.patch_size

        if voxel_tokens.shape[0] == 0:
            empty_long = torch.zeros((0, self.cam_num), dtype=torch.long, device=points.device)
            empty_bool = torch.zeros((0, self.cam_num), dtype=torch.bool, device=points.device)
            return {
                "voxel_tokens": voxel_tokens,
                "voxel_batch_ids": torch.zeros((0,), dtype=torch.long, device=points.device),
                "patch_y": empty_long,
                "patch_x": empty_long,
                "visible": empty_bool,
                "patch_h": patch_h,
                "patch_w": patch_w,
            }

        voxel_centers = self._voxel_coords_to_centers(voxel_coords)
        voxel_batch_ids = voxel_coords[:, 0].long()
        if lidar_to_world is not None:
            voxel_centers = self._local_voxel_centers_to_world(voxel_centers, voxel_batch_ids, lidar_to_world)
        patch_y, patch_x, visible = self._project_voxels_to_patches(
            voxel_centers,
            voxel_batch_ids,
            intrinsics,
            camera_to_world,
            patch_h,
            patch_w,
        )
        return {
            "voxel_tokens": voxel_tokens,
            "voxel_batch_ids": voxel_batch_ids,
            "patch_y": patch_y,
            "patch_x": patch_x,
            "visible": visible,
            "patch_h": patch_h,
            "patch_w": patch_w,
        }

    def _mix_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        fusion_data: Dict[str, torch.Tensor | int],
        layer_idx: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, patch_count, channels = patch_tokens.shape
        frame_count = sequence_length // self.cam_num
        patch_h = int(fusion_data["patch_h"])
        patch_w = int(fusion_data["patch_w"])
        if patch_h * patch_w != patch_count:
            raise ValueError(
                f"Patch count mismatch: expected {patch_h * patch_w} from patch grid "
                f"{patch_h}x{patch_w}, but got {patch_count}"
            )

        fused_tokens = patch_tokens.reshape(batch_size, frame_count, self.cam_num, patch_count, channels).clone()
        voxel_tokens = fusion_data["voxel_tokens"]
        voxel_batch_ids = fusion_data["voxel_batch_ids"]
        patch_y = fusion_data["patch_y"]
        patch_x = fusion_data["patch_x"]
        visible = fusion_data["visible"]

        for batch_idx in range(batch_size):
            for frame_idx in range(frame_count):
                flat_frame_idx = batch_idx * frame_count + frame_idx
                frame_mask = voxel_batch_ids == flat_frame_idx

                frame_voxel_tokens = voxel_tokens[frame_mask]
                frame_patch_y = patch_y[frame_mask]
                frame_patch_x = patch_x[frame_mask]
                frame_visible = visible[frame_mask]

                for cam_idx in range(self.cam_num):
                    cam_mask = frame_visible[:, cam_idx]
                    view_voxel_tokens = frame_voxel_tokens[cam_mask]
                    view_patch_y = frame_patch_y[cam_mask, cam_idx]
                    view_patch_x = frame_patch_x[cam_mask, cam_idx]
                    fused_tokens[batch_idx, frame_idx, cam_idx] = self.window_fusions[layer_idx](
                        fused_tokens[batch_idx, frame_idx, cam_idx],
                        view_voxel_tokens,
                        view_patch_y,
                        view_patch_x,
                        patch_h,
                        patch_w,
                    )

        return fused_tokens.reshape(batch_size, sequence_length, patch_count, channels)
