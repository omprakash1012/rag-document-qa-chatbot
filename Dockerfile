# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# No index is baked into the image on purpose (docs/ change per deployment).
# Run `docker compose run rag-api python ingest.py` once before serving,
# or mount a pre-built vectorstore/ volume (see docker-compose.yml).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
