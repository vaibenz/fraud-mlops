FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/models/champion PREDICTION_LOG=/app/logs/predictions.jsonl

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY fraud/ fraud/
COPY models/champion/ models/champion/

RUN useradd --create-home appuser && mkdir -p /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fraud.serve:app", "--host", "0.0.0.0", "--port", "8000"]
