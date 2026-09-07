FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel@sha256:024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1

WORKDIR /app

# The base image's retired NVIDIA apt sources make apt-get update fail. GraspNet only needs
# build tools from Ubuntu to compile PointNet2 and KNN. libsuitesparse-dev provides umfpack.h —
# without it, cvxopt (graspnetAPI's dependency) fails to build its optional UMFPACK extension
# (2026-09-05 실물 PC에서 확인: "fatal error: umfpack.h: No such file or directory"). libgl1/
# libglib2.0-0은 open3d(graspnetAPI 의존)가 import 시점에 dlopen하는 libGL.so.1을 위한 것 —
# perception 이미지가 opencv-python 때문에 같은 라이브러리를 이미 넣고 있는 것과 같은 이유다.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git build-essential liblapack-dev libblas-dev libsuitesparse-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# `docker build`에는 GPU가 안 보인다 — torch cpp_extension이 실행 중인 디바이스에서 arch를
# 알아내려다(_get_cuda_arch_flags) 못 찾으면 빈 리스트에 인덱싱해 죽는다(2026-09-05 실물 PC:
# "IndexError: list index out of range"). 이 이미지의 nvcc(CUDA 11.1)가 지원하는 범위 안에서
# 명시하고, 최고 arch에 PTX를 얹어 이후 세대 GPU(예: Ada, sm_89)에서도 드라이버가 JIT해
# 돌아가게 한다 — 이 테스트 PC의 RTX 4060(sm_89)이 정확히 그 경우다.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

# ruamel.yaml: graspnetAPI → autolab_core가 import 시점에 필요로 하는데(yaml_config.py),
# base conda 이미지에 이미 distutils로 깔려 있는 ruamel-yaml 0.15.87이 import이 안 되는
# (혹은 깨진) 상태였다 — pip이 RECORD 없는 distutils 설치를 못 지워 그냥 넘어간 것으로
# 보인다(2026-09-05 실물 PC 확인: "Cannot uninstall 'ruamel-yaml' ... distutils installed").
# web/perception Dockerfile이 apt python3-numpy에 쓰는 것과 같은 우회(--ignore-installed).
#
# psutil은 지우고 새로 깐다 — base 이미지의 conda psutil 5.8.0이 남긴 구 C 확장
# (_psutil_linux.cpython-38-*.so)이 CPython 3.8 확장 검색 순서상 pip이 새로 받은
# 7.x의 _psutil_linux.abi3.so보다 먼저 잡혀 "no attribute 'getpagesize'"로 깨졌다
# (open3d.ml → sklearn → joblib → psutil 체인, 2026-09-05 실물 PC 확인).
# --force-reinstall만으로는 conda가 깐 파일이 안 지워져 두 버전이 뒤섞인 채 남는다.
RUN git clone https://github.com/graspnet/graspnet-baseline.git /opt/graspnet-baseline \
    && git -C /opt/graspnet-baseline checkout --detach 280c215129f759ed8649cb4e89fc5dfee55f4f80 \
    && git clone https://github.com/graspnet/graspnetAPI.git /opt/graspnetAPI \
    && git -C /opt/graspnetAPI checkout --detach eb57dd2092d8dbe05312a29c3d0c22f3226efbfc \
    && pip install --no-cache-dir --ignore-installed -e /opt/graspnetAPI open3d==0.13.0 scipy \
        "ruamel.yaml>=0.15.94" \
    && rm -rf /opt/conda/lib/python3.8/site-packages/psutil* \
    && pip install --no-cache-dir --force-reinstall psutil \
    && cd /opt/graspnet-baseline/pointnet2 && python setup.py install \
    && cd /opt/graspnet-baseline/knn && python setup.py install

COPY runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
