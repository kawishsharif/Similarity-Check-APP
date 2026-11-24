FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libreoffice-writer libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
      fonts-noto-core fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /data /srv
USER app
ENV DATA_DIR=/data BRAND_FILE=/data/brand.yaml
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
