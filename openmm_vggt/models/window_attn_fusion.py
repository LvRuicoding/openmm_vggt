from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
