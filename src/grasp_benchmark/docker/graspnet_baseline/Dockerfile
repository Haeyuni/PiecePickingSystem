FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
WORKDIR /app
# This base image bakes in an NVIDIA cuda apt source whose signing key was rotated out; its
# repo now 404s the key and breaks `apt-get update`. We don't need packages from it (only
# git/build-essential from the Ubuntu archive), so drop it before updating.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/graspnet/graspnet-baseline.git /opt/graspnet-baseline \
    && git clone --depth 1 https://github.com/graspnet/graspnetAPI.git /opt/graspnetAPI \
    && pip install --no-cache-dir -e /opt/graspnetAPI open3d==0.13.0 gdown scipy \
    && cd /opt/graspnet-baseline/pointnet2 && python setup.py install \
    && cd /opt/graspnet-baseline/knn && python setup.py install
COPY docker/common /app/common
COPY docker/graspnet_baseline/runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
