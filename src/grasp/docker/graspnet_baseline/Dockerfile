FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel@sha256:024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1

WORKDIR /app

# The base image's retired NVIDIA apt sources make apt-get update fail. GraspNet only needs
# build tools from Ubuntu to compile PointNet2 and KNN.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends git build-essential liblapack-dev libblas-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/graspnet/graspnet-baseline.git /opt/graspnet-baseline \
    && git -C /opt/graspnet-baseline checkout --detach 280c215129f759ed8649cb4e89fc5dfee55f4f80 \
    && git clone https://github.com/graspnet/graspnetAPI.git /opt/graspnetAPI \
    && git -C /opt/graspnetAPI checkout --detach eb57dd2092d8dbe05312a29c3d0c22f3226efbfc \
    && pip install --no-cache-dir -e /opt/graspnetAPI open3d==0.13.0 scipy \
    && cd /opt/graspnet-baseline/pointnet2 && python setup.py install \
    && cd /opt/graspnet-baseline/knn && python setup.py install

COPY runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
