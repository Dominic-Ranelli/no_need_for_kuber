# Version
FROM python:3.11-slim

# Env variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=80

WORKDIR /code

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and needed artifacts
COPY ./app ./app
COPY ./models ./models
COPY ./data ./data
COPY CONFIGS.yaml .

# Non-root for security
RUN useradd -m appuser && chown -R appuser:appuser /code
USER appuser

# Port
EXPOSE 80

# Running
CMD ["fastapi", "run", "app/main.py", "--proxy-headers", "--port", "80", "--host", "0.0.0.0"]
