"""CLIP scene encoding utilities for the paper's 112x112 Scene Stream."""

import math

import torch
import torch.nn.functional as F


def encode_with_interpolated_positions(image_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return pooled CLIP features at arbitrary patch-grid resolutions.

    Transformers 4.41 fixes CLIP's learned positional table to its pretraining
    resolution. Eco-Mamba uses 112x112 frames, so the patch portion of that
    table must be interpolated before running the frozen vision transformer.
    """
    vision = image_model.vision_model
    embeddings = vision.embeddings
    target_dtype = embeddings.patch_embedding.weight.dtype
    patches = embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
    grid_h, grid_w = patches.shape[-2:]
    patches = patches.flatten(2).transpose(1, 2)
    class_token = embeddings.class_embedding.expand(patches.shape[0], 1, -1)

    positions = embeddings.position_embedding.weight
    class_position = positions[:1].unsqueeze(0)
    patch_positions = positions[1:]
    source_grid = math.isqrt(patch_positions.shape[0])
    if source_grid * source_grid != patch_positions.shape[0]:
        raise ValueError("CLIP patch positional embeddings must form a square grid")
    patch_positions = patch_positions.reshape(
        1, source_grid, source_grid, -1
    ).permute(0, 3, 1, 2)
    patch_positions = F.interpolate(
        patch_positions, size=(grid_h, grid_w), mode="bicubic", align_corners=False
    ).permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)

    hidden_states = torch.cat([class_token, patches], dim=1)
    hidden_states = hidden_states + torch.cat([class_position, patch_positions], dim=1)
    hidden_states = vision.pre_layrnorm(hidden_states)
    encoded = vision.encoder(inputs_embeds=hidden_states, return_dict=True).last_hidden_state
    return vision.post_layernorm(encoded[:, 0])
