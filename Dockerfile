# The environment is packed from the verified roadside_perception Conda environment.
FROM ubuntu:22.04 AS env-unpack
COPY roadside_perception-env.tar.gz /tmp/roadside_perception-env.tar.gz
RUN mkdir -p /opt/roadside-env \
    && tar -xzf /tmp/roadside_perception-env.tar.gz -C /opt/roadside-env \
    && PATH=/opt/roadside-env/bin:$PATH /opt/roadside-env/bin/conda-unpack

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/roadside-env/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    WANDB_MODE=disabled \
    XDG_CACHE_HOME=/tmp/.cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=env-unpack /opt/roadside-env /opt/roadside-env
WORKDIR /app
COPY algorithms/rddetr ./algorithms/rddetr
COPY algorithms/stmc ./algorithms/stmc
COPY configs ./configs
COPY models ./models
COPY pipeline ./pipeline
COPY scripts ./scripts
COPY schemas ./schemas
COPY docs ./docs
COPY examples ./examples
COPY third_party/HyperLPR/Prj-Python/hyperlpr3 ./third_party/HyperLPR/Prj-Python/hyperlpr3
COPY third_party/HyperLPR/resource/font/platech.ttf ./third_party/HyperLPR/resource/font/platech.ttf
COPY README.md SOURCE_MANIFEST.md environment.yml requirements.pipeline.txt ./
ENTRYPOINT ["python", "scripts/run_from_directory.py"]
