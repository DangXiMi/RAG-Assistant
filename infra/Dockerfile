FROM python:3.11-slim
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --no-deps -r requirements.lock
COPY src/ ./src/
CMD ["python", "-m", "src.main"]