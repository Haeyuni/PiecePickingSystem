FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY docker/common /app/common
COPY docker/pca_normal/runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
