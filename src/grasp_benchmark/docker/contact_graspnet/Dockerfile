FROM tensorflow/tensorflow:2.2.0-gpu-py3
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy==1.19.2 scipy==1.4.1 opencv-python==4.4.0.46 trimesh==3.8.12 gdown==4.7.1
RUN git clone --depth 1 https://github.com/NVlabs/contact_graspnet.git /opt/contact_graspnet \
    && cd /opt/contact_graspnet && sh compile_pointnet_tfops.sh
COPY docker/common /app/common
COPY docker/contact_graspnet/runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
