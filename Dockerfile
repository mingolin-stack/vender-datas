# 判讀 API 部署到 Google Cloud Run 用
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY api.py extract_core.py auto_align.py vision_utils.py ./
COPY templates ./templates
ENV PORT=8080
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
