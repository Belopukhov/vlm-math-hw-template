# Домашнее задание: VLM д

В этом проекте нужно реализовать упрощённый пайплайн VLM: изображение с графиком, схемой, таблицей или геометрической фигурой + текстовый вопрос -> текстовый ответ.

Задание специально разделено на три трека, чтобы его можно было честно сдавать при разном доступе к вычислительны мощностям:

| Трек | Ресурс у студента | Что обязательно | Что не обязательно |
|---|---:|---|---|
| **A. CPU-only** | GPU нет | Реализовать код, пройти unit/smoke tests | Получить высокое качество |
| **B. Small GPU** | 6–12 GB VRAM | Adapter-only обучение на маленьком math subset | LoRA и большой benchmark |
| **C. A100-20GB** | 1/4 A100, около 20 GB VRAM | Adapter pretrain + SFT с LoRA | Rank 256 и тяжёлый leaderboard |

Основная оценка ставится за корректный инженерный пайплайн. Качество на hidden math benchmark используется только для расширенного трека или бонуса.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q tests_public
```

Для CPU-трека достаточно добиться прохождения public-тестов и написать короткий отчёт.

## Что нужно реализовать

Файлы с `TODO` находятся в папке `hw/`:

```text
hw/dataset.py      # загрузка math-VQA примеров
hw/processor.py    # preprocessing изображений, prompt, labels, collate
hw/model.py        # adapter, visual-token merge, forward/generate
hw/train.py        # training loop, сохранение adapter/checkpoint
hw/benchmark.py    # prompt для benchmark, parse ответа, accuracy
```

Запрещено менять интерфейсы функций и классов, которые используются в `tests_public/`.

## Данные

В репозитории есть маленький toy-набор:

```text
assets/toy_math_vqa/
  manifest.jsonl
  images/*.png
```

Он нужен для public-тестов и smoke-запусков. 

## Команды по трекам

### Track A: CPU-only

```bash
pytest -q tests_public
python -m hw.train --config configs/track_a_cpu.yaml --fast-train
python -m hw.benchmark --config configs/inference_math.yaml --toy
```

### Track B: Small GPU

```bash
python -m hw.train --config configs/track_b_small_gpu.yaml
python -m hw.benchmark --config configs/inference_math.yaml
```

### Track C: A100-20GB

```bash
python -m hw.train --config configs/track_c_a100_pretrain.yaml
python -m hw.train --config configs/track_c_a100_sft.yaml
python -m hw.benchmark --config configs/inference_math.yaml
```

## Что сдавать

План минимум:

```text
hw/*.py
report.md
```

Для GPU-треков дополнительно:

```text
artifacts/adapter.pt или artifacts/adapter.safetensors
artifacts/special_tokens.pt, если вы обучали новые visual special tokens
artifacts/lora или artifacts/model.pt, если вы делали SFT с LoRA
```

Не добавляйте в репозиторий большие файлы без разрешения преподавателя. Для больших чекпойнтов используйте место сдачи, указанное в LMS.
