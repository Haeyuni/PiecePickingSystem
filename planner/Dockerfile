# TODO: 베이스 이미지는 docs/environment.md 확정 후 반영
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# TODO: entrypoint 구성 (uvicorn 실행)
