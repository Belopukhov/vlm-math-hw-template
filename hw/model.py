from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space.

    Pipeline:
        [B, N, vision_hidden_size]
            -> LayerNorm
            -> Linear (vision_hidden_size -> text_hidden_size)
            -> GELU
            -> Linear (text_hidden_size -> text_hidden_size)
            -> adaptive pooling to num_image_tokens along the sequence axis.
        => [B, num_image_tokens, text_hidden_size]
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.norm(vision_hidden_states)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)

        b, n, d = x.shape
        if n == self.num_image_tokens:
            return x

        x_t = x.transpose(1, 2)
        x_t = nn.functional.adaptive_avg_pool1d(x_t, self.num_image_tokens)
        return x_t.transpose(1, 2).contiguous()


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    if input_embeds.shape[0] != visual_embeds.shape[0]:
        raise ValueError(
            f"Batch size mismatch: input_embeds={input_embeds.shape[0]} "
            f"vs visual_embeds={visual_embeds.shape[0]}"
        )

    merged = input_embeds.clone()
    bsz, k, dim = visual_embeds.shape

    for b in range(bsz):
        mask = input_ids[b] == image_token_id
        positions = mask.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        n = min(positions.numel(), k)
        merged[b, positions[:n]] = visual_embeds[b, :n].to(merged.dtype)
    return merged


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision encoder, flattening the tile axis into the batch.

        Input: [B, T, 3, H, W]
        Output: [B, K, text_hidden_size] visual embeddings ready to be inserted.
        """
        b, t, c, h, w = pixel_values.shape
        flat = pixel_values.view(b * t, c, h, w)

        out = self.vision_encoder(flat)
        hidden = self._extract_hidden(out)

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)

        bt, n, d = hidden.shape
        hidden = hidden.view(b, t * n, d)

        visual = self.adapter(hidden)
        return visual

    @staticmethod
    def _extract_hidden(out: Any) -> torch.Tensor:
        """Best-effort extraction of hidden states from various encoder outputs."""
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("last_hidden_state", "hidden_states", "pooler_output"):
            value = getattr(out, attr, None)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    value = value[-1]
                return value
        if isinstance(out, dict):
            for key in ("last_hidden_state", "hidden_states", "pooler_output"):
                if key in out:
                    val = out[key]
                    if isinstance(val, (list, tuple)):
                        val = val[-1]
                    return val
        raise TypeError(f"Cannot extract hidden states from vision encoder output: {type(out)}")

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        get_emb = getattr(self.language_model, "get_input_embeddings", None)
        if get_emb is not None:
            return get_emb()(input_ids)
        if hasattr(self.language_model, "embed_tokens"):
            return self.language_model.embed_tokens(input_ids)
        raise AttributeError("language_model has no input embedding layer")

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        pixel_values = batch["pixel_values"]

        visual = self._encode_images(pixel_values)
        text_embeds = self._text_embeddings(input_ids)
        inputs_embeds = merge_visual_embeddings(
            text_embeds, input_ids, visual, self.config.image_token_id
        )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        pixel_values = batch["pixel_values"]

        visual = self._encode_images(pixel_values)
        text_embeds = self._text_embeddings(input_ids)
        inputs_embeds = merge_visual_embeddings(
            text_embeds, input_ids, visual, self.config.image_token_id
        )

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
