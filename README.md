# License Plate Detection System

**Автор:** Анастасия Панчева

## Результаты модели

| Метрика | Значение |
|---------|----------|
| **mAP50** | **0.935 (93.5%)** |
| mAP50-95 | 0.671 |
| Precision | 1.000 (100%) |
| Recall | 0.933 |

**S-Tier результат** (>0.8 mAP)

## Быстрый старт

```bash
# Установка
poetry install

# Запуск
python src/main.py --mode video --input video.mp4 --output result.mp4 --device cpu

# С OCR
python src/main.py --mode video --input video.mp4 --ocr --device cpu

# С определением скорости
python src/main.py --mode video --input video.mp4 --speed --device cpu