# Report

## Track

A (CPU-only)

## Что реализовано

- [x] `hw/dataset.py`
- [x] `hw/processor.py`
- [x] `hw/model.py`
- [x] `hw/train.py`
- [x] `hw/benchmark.py`

## Конфигурация

```text
config: configs/track_a_cpu.yaml
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: 1
```

## Результаты

```text
public tests: 14 passed
train loss:   6.51 -> 6.01 (2 шага fast-train)
benchmark:    overall=0.25 (toy, baseline)
```

## Запуск

```bash
pip install -e ".[dev]"
pytest -q tests_public
python -m hw.train --config configs/track_a_cpu.yaml --fast-train
python -m hw.benchmark --config configs/inference_math.yaml --toy
```

## Ресурсы

CPU, без GPU. Обучение ~3 сек, тесты ~3 сек.

## Анализ ошибок

Качество по этому набору не оценивается.

## Критерии оценивания

См. [`GRADING.md`](GRADING.md).
