FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app
COPY . .

RUN poetry install --no-root

ENTRYPOINT ["poetry", "run", "python", "src/main.py"]