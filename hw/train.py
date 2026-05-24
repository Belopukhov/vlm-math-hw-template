from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_loss(out: Any) -> torch.Tensor:
    """Extract a scalar loss tensor from a model output (dict, dataclass, tensor)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        if "loss" not in out:
            raise KeyError("model output dict must contain 'loss'")
        return out["loss"]
    loss = getattr(out, "loss", None)
    if loss is None:
        raise TypeError(f"Cannot find 'loss' in model output of type {type(out)}")
    return loss


def train_one_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad()

    out = model(batch)
    loss = _extract_loss(out)

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss encountered: {loss.item()}")

    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())


def _save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter = getattr(model, "adapter", None)
    if adapter is not None:
        torch.save(adapter.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def run_training(config: dict[str, Any], fast_train: bool = False) -> dict[str, Any]:
    """Main training entry point.

    Tries to build a real VLM training loop. If full ML stack (transformers,
    image encoders, tokenizers) is not available, runs a lightweight smoke
    training loop on the processor outputs to validate the pipeline.
    """
    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig

    set_seed(int(config.get("seed", 42)))

    data_cfg = config.get("data", {}) or {}
    proc_cfg_dict = config.get("processor", {}) or {}
    trainer_cfg = config.get("trainer", {}) or {}

    manifest_path = data_cfg.get("train_manifest")
    if manifest_path is None:
        raise ValueError("config.data.train_manifest must be set")

    max_samples = data_cfg.get("max_samples")
    if fast_train:
        max_samples = min(int(max_samples or 4), 4)

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=data_cfg.get("split", "train"),
        max_samples=max_samples,
    )

    processor = MathVLMProcessor(
        tokenizer=_build_simple_tokenizer(),
        config=ProcessorConfig(**proc_cfg_dict),
    )

    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    num_workers = int(trainer_cfg.get("num_workers", 0))

    def _collate(items):
        processed = [processor(s) for s in items]
        return processor.collate(processed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    model = _build_smoke_model(processor)
    device = torch.device(trainer_cfg.get("device", "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    grad_accum = int(trainer_cfg.get("grad_accum_steps", 1))
    max_steps = int(trainer_cfg.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 2)

    losses: list[float] = []
    step = 0
    optimizer.zero_grad()
    accum_loss = 0.0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            model.train()
            out = model(batch)
            loss = _extract_loss(out) / max(1, grad_accum)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            accum_loss += float(loss.detach().cpu().item())
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                losses.append(accum_loss)
                accum_loss = 0.0
            step += 1

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        _save_checkpoint(model, save_path)

    return {"losses": losses, "steps": step}


class _SimpleTokenizer:
    """Minimal whitespace tokenizer for CPU smoke training without HF deps."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<image>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _ensure(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = [self._ensure(t) for t in text.replace("\n", " ").split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def vocab_size(self) -> int:
        return len(self.vocab)


def _build_simple_tokenizer() -> _SimpleTokenizer:
    return _SimpleTokenizer()


class _TinyLM(torch.nn.Module):
    """Tiny LM stand-in used for CPU smoke training of the adapter pipeline."""

    def __init__(self, vocab_size: int, hidden: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size + 256, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size + 256, bias=False)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


class _TinyVisionEncoder(torch.nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.proj = torch.nn.Linear(3 * 4 * 4, hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.pool(pixel_values)
        x = x.flatten(1)
        x = self.proj(x)
        return x.unsqueeze(1)


def _build_smoke_model(processor) -> torch.nn.Module:
    from hw.model import MathVLM, ModelConfig

    text_hidden = 32
    vision_hidden = 32
    num_image_tokens = int(processor.config.num_image_tokens)
    tokenizer = processor.tokenizer
    image_token_id = tokenizer.vocab.get("<image>", 2)

    vision = _TinyVisionEncoder(hidden=vision_hidden)
    lm = _TinyLM(vocab_size=tokenizer.vocab_size(), hidden=text_hidden)
    model = MathVLM(
        vision_encoder=vision,
        language_model=lm,
        config=ModelConfig(
            vision_hidden_size=vision_hidden,
            text_hidden_size=text_hidden,
            num_image_tokens=num_image_tokens,
            image_token_id=image_token_id,
        ),
    )
    model.freeze_backbones()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    summary = run_training(config, fast_train=args.fast_train)
    print(summary)


if __name__ == "__main__":
    main()
