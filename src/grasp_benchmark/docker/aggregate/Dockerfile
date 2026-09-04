FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas openpyxl pillow
COPY docker/aggregate/runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
