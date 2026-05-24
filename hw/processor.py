from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


def _resize_with_pad(image: Image.Image, target_size: int) -> Image.Image:
    """Resize keeping aspect ratio, then pad to a square `target_size` canvas."""
    w, h = image.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), (0, 0, 0))
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to normalized [3, H, W] float tensor."""
    raw = bytearray(image.tobytes())
    arr = torch.frombuffer(raw, dtype=torch.uint8)
    arr = arr.view(image.size[1], image.size[0], 3)
    tensor = arr.permute(2, 0, 1).contiguous().float().div_(255.0)
    mean = torch.tensor(_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        size = int(self.config.image_size)
        num_tiles = max(1, int(self.config.num_tiles))

        canvas = _resize_with_pad(image, size)

        if num_tiles == 1:
            tile_tensor = _image_to_tensor(canvas).unsqueeze(0)
            return tile_tensor

        grid = max(1, int(round(num_tiles**0.5)))
        if grid * grid == num_tiles:
            rows, cols = grid, grid
        else:
            rows, cols = 1, num_tiles

        tile_w = size // cols
        tile_h = size // rows
        tiles: list[torch.Tensor] = []
        for r in range(rows):
            for c in range(cols):
                left = c * tile_w
                upper = r * tile_h
                right = size if c == cols - 1 else left + tile_w
                lower = size if r == rows - 1 else upper + tile_h
                crop = canvas.crop((left, upper, right, lower)).resize((size, size), Image.BILINEAR)
                tiles.append(_image_to_tensor(crop))

        while len(tiles) < num_tiles:
            tiles.append(_image_to_tensor(canvas))
        tiles = tiles[:num_tiles]

        return torch.stack(tiles, dim=0)

    def _image_placeholder(self) -> str:
        n = max(1, int(self.config.num_image_tokens))
        return f"{IMAGE_START_TOKEN} " + (f"{IMAGE_TOKEN} " * n) + IMAGE_END_TOKEN

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options."""
        options_text = "\n".join(sample.options)
        prompt = (
            f"{self._image_placeholder()}\n"
            "Реши визуально-математическую задачу. "
            "Выбери один вариант ответа и в конце напиши только букву.\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            "Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def _tokenize(self, text: str) -> list[int]:
        out = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(self.config.max_length),
        )
        if isinstance(out, dict):
            ids = out["input_ids"]
        else:
            ids = out
        return list(ids)

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample."""
        prompt_only = self.build_prompt(sample, include_answer=False)
        full = self.build_prompt(sample, include_answer=True)

        prompt_ids = self._tokenize(prompt_only)
        full_ids = self._tokenize(full)

        if full_ids[: len(prompt_ids)] != prompt_ids:
            answer_ids = self._tokenize(sample.answer)
            full_ids = prompt_ids + answer_ids

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
            full_ids = full_ids + [int(eos_id)]

        max_len = int(self.config.max_length)
        full_ids = full_ids[:max_len]

        n_prompt = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = int(self.config.ignore_index)

        if all(l == self.config.ignore_index for l in labels) and len(full_ids) > n_prompt:
            for j in range(n_prompt, len(full_ids)):
                labels[j] = int(full_ids[j])

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels_t = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_t,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values."""
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        pad_id = int(pad_id)
        ignore_index = int(self.config.ignore_index)

        max_len = max(int(item["input_ids"].shape[0]) for item in batch)

        input_ids_list: list[torch.Tensor] = []
        attn_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        pixel_list: list[torch.Tensor] = []

        for item in batch:
            ids = item["input_ids"]
            attn = item["attention_mask"]
            labels = item["labels"]
            pad_len = max_len - int(ids.shape[0])
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)], dim=0)
                attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)], dim=0)
                labels = torch.cat(
                    [labels, torch.full((pad_len,), ignore_index, dtype=labels.dtype)],
                    dim=0,
                )
            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(labels)
            pixel_list.append(item["pixel_values"])

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attn_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
            "pixel_values": torch.stack(pixel_list, dim=0),
        }
        return out
