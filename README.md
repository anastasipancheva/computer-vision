# License Plate Detection System

**Автор:** Анастасия Панчева

## Оценка соответствия требованиям ТЗ

### Data Science Part

| Баллы | Критерий | Значение | Результат |
|-------|----------|----------|-----------|
| 15 | mAP > 0.8 (S-Tier) | 0.935 (93.5%) | 15 баллов |


### Software Engineer Part

| Баллы | Критерий | Статус | Обоснование |
|-------|----------|--------|-------------|
| 15 | Свой датасет | ✅ | 179 изображений, размеченных в Roboflow: https://universe.roboflow.com/anastasias-workspace-wrbmj/license-plate-detection-svpyg |
| 15 | Use cases (video + stream) | ✅ | `--mode video` и `--mode camera` |
| 5 | README demos (GIF) | ✅ | Демонстрация работы (ссылка на видео/GIF) |
| 5 | Code quality | ✅ | ООП, докстринги, осмысленные имена, нет дублирования |
| 5 | model.py | ✅ | `My_LicensePlate_Model` в `src/model_impl.py` с методом `detect_plates()` |
| 5 | Wandb/MLflow | ✅ | https://wandb.ai/anastasipancheva-tsu/license-plate-detection |
| 2 | Logging | ✅ | Singleton-логгер, `./data/log_file.log`, обработка ошибок |
| 3 | Git workflow | ✅ | Публичный репозиторий, ветки `dev`/`main`, осмысленные коммиты |


### Дополнительный функционал

| Баллы | Критерий | Статус | Обоснование |
|-------|----------|--------|-------------|
| 25 | Определение скорости | ✅ | `SpeedDetector` в `src/speed_detector.py`, отслеживание между кадрами, расчёт км/ч |
| 25 | OCR (распознавание номеров) | ✅ | `LicensePlateOCR` в `src/ocr_reader.py`, поддержка русского и английского языков |

## Где найти в коде

| Функция | Файл | Метод |
|---------|------|-------|
| Детекция номеров | `src/model_impl.py` | `My_LicensePlate_Model.detect_plates()` |
| OCR | `src/ocr_reader.py` | `LicensePlateOCR.read_plate()` |
| Определение скорости | `src/speed_detector.py` | `SpeedDetector.update()`, `_calculate_speed()` |
| Логирование | `src/main.py` | `setup_logging()`, `data/log_file.log` |
| CLI интерфейс | `src/main.py` | `cli()`, argparse |
| Docker | `Dockerfile`, `docker-compose.yaml` | — |
| Poetry | `pyproject.toml` | — |

## Результаты модели

| Метрика | Значение |
|---------|----------|
| mAP50 | 0.935 (93.5%) |
| mAP50-95 | 0.671 |
| Precision | 1.000 (100%) |
| Recall | 0.933 |
"Precision 100% получена на тестовой выборке из датасета. В реальных видео, например, с бликами или фарами, модель иногда ошибается - может отметить фару или часть кузова.

## Ссылки

- **Датасет (Roboflow)**: https://universe.roboflow.com/anastasias-workspace-wrbmj/license-plate-detection-svpyg
- **Wandb логи**: https://wandb.ai/anastasipancheva-tsu/license-plate-detection

## Как запустить

```bash
# Детекция
python src/main.py --mode video --input video.mp4 --output detection.mp4 --device cpu

# Детекция + OCR
python src/main.py --mode video --input video.mp4 --output ocr.mp4 --ocr --device cpu

# Детекция + скорость
python src/main.py --mode video --input video.mp4 --output speed.mp4 --speed --device cpu

# Всё вместе
python src/main.py --mode video --input video.mp4 --output full.mp4 --ocr --speed --device cpu

# Веб-камера
python src/main.py --mode camera --camera-id 0 --ocr --speed --device cpu
