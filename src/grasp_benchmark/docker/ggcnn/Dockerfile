FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git wget unzip && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir 'numpy<2' scikit-image
RUN git clone --depth 1 https://github.com/dougsm/ggcnn.git /opt/ggcnn
COPY docker/common /app/common
COPY docker/ggcnn/runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
