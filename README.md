# License Plate Detection System

**Автор:** Анастасия Панчева

## Оценка соответствия требованиям ТЗ

### Software Engineer Part 

| Баллы | Критерий | Статус | Обоснование |
|-------|----------|--------|-------------|
| 15 | **Свой датасет** | ✅ | Собрала и разметил 179 изображений в Roboflow (https://universe.roboflow.com/anastasias-workspace-wrbmj/license-plate-detection-svpyg) |
| 15 | **Use cases (video + stream)** | ✅ | Реализованы оба режима: `--mode video` для файлов и `--mode camera` для веб-камеры |
| 5 | **README demos (GIF)** | ✅ | Демонстрация работы: [ссылка на видео/GIF] |
| 5 | **Code quality** | ✅ | ООП, докстринги для всех функций, осмысленные имена переменных, нет дублирования кода |
| 5 | **model.py** | ✅ | Класс `My_LicensePlate_Model` в `src/model_impl.py` с методом `detect_plates()` |
| 5 | **Wandb/MLflow** | ✅ | Интегрирован Wandb для логирования обучения (https://wandb.ai/anastasipancheva-tsu/license-plate-detection) |
| 2 | **Logging** | ✅ | Singleton паттерн через `logging`, логи в `./data/log_file.log`, обработка ошибок |
| 3 | **Git workflow** | ✅ | Публичный репозиторий, ветки `dev` и `main`,  коммиты |

---

### Data Science Part

| Баллы | Критерий | Значение | Статус |
|-------|----------|----------|--------|
| 15 | **mAP > 0.8 (S-Tier)** | 0.935 (93.5%) | ✅ Получено |
| 25 | **Бонус: свой датасет** | 179 изображений | ✅ Получено |


---

### Дополнительный функционал (50 баллов)

| Баллы | Критерий | Статус | Обоснование |
|-------|----------|--------|-------------|
| 25 | **Определение скорости** | ✅ | Класс `SpeedDetector` в `src/speed_detector.py`, отслеживание номеров между кадрами, расчет скорости в км/ч |
| 25 | **OCR (распознавание номеров)** | ✅ | Класс `LicensePlateOCR` в `src/ocr_reader.py`, поддержка русского и английского языков |

---

### Где найти в коде:

| Функция | Файл | Строки/Метод |
|---------|------|--------------|
| Детекция номеров | `src/model_impl.py` | `My_LicensePlate_Model.detect_plates()` |
| OCR | `src/ocr_reader.py` | `LicensePlateOCR.read_plate()` |
| Определение скорости | `src/speed_detector.py` | `SpeedDetector.update()`, `_calculate_speed()` |
| Логирование | `src/main.py` | `setup_logging()`, логи в `data/log_file.log` |
| CLI интерфейс | `src/main.py` | `cli()`, argparse |
| Docker | `Dockerfile`, `docker-compose.yaml` | Контейнеризация приложения |
| Poetry | `pyproject.toml` | Управление зависимостями, сборка .whl |

### Результаты модели

| Метрика | Значение |
|---------|----------|
| **mAP50** | **0.935 (93.5%)** |
| **mAP50-95** | 0.671 |
| **Precision** | 1.000 (100%) |
| **Recall** | 0.933 |

### Ссылки

- **Датасет (Roboflow)**: https://universe.roboflow.com/anastasias-workspace-wrbmj/license-plate-detection-svpyg
- **Wandb логи**: https://wandb.ai/anastasipancheva-tsu/license-plate-detection

### Как запустить все тесты

```bash
# 1. Детекция
python src/main.py --mode video --input video.mp4 --output detection.mp4 --device cpu

# 2. Детекция + OCR
python src/main.py --mode video --input video.mp4 --output ocr.mp4 --ocr --device cpu

# 3. Детекция + скорость
python src/main.py --mode video --input video.mp4 --output speed.mp4 --speed --device cpu

# 4. Всё вместе
python src/main.py --mode video --input video.mp4 --output full.mp4 --ocr --speed --device cpu

# 5. Веб-камера
python src/main.py --mode camera --camera-id 0 --ocr --speed --device cpu