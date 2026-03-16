# syntax=docker/dockerfile:1.4
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

# ── Version pins (bump these ARGs to update) ─────────────────────────
ARG LEVEL_ZERO_VERSION=1.28.0
ARG IGC_VERSION=2.28.4
ARG IGC_BUILD=20760
ARG COMPUTE_RUNTIME_VERSION=26.05.37020.3
ARG GMMLIB_VERSION=22.9.0
ARG IPEXLLM_BUNDLE=ollama-ipex-llm-2.3.0b20250725-ubuntu.tgz

# ── Base packages ─────────────────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install --no-install-recommends -q -y \
      software-properties-common \
      ca-certificates \
      wget \
      ocl-icd-libopencl1

# ── Intel GPU runtimes ────────────────────────────────────────────────
# Sources: compute-runtime, level-zero, IGC — see release notes at
#   https://github.com/intel/compute-runtime/releases/tag/${COMPUTE_RUNTIME_VERSION}
RUN --mount=type=cache,target=/tmp/dl-gpu \
    cd /tmp/dl-gpu && \
    wget -nc https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb && \
    wget -nc https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget -nc https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget -nc https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc-dbgsym_${COMPUTE_RUNTIME_VERSION}-0_amd64.ddeb && \
    wget -nc https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget -nc https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget -nc https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMMLIB_VERSION}_amd64.deb && \
    wget -nc https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    mkdir -p /tmp/gpu-install && cp *.deb *.ddeb /tmp/gpu-install/ && \
    cd /tmp/gpu-install && dpkg -i *.deb *.ddeb && \
    rm -rf /tmp/gpu-install

# ── IPEX-LLM portable zip (ollama bundle, ollama v0.9.3) ─────────────
# https://github.com/ipex-llm/ipex-llm/releases/tag/v2.3.0-nightly
RUN --mount=type=cache,target=/tmp/dl-ipex \
    cd /tmp/dl-ipex && \
    wget -nc https://github.com/ipex-llm/ipex-llm/releases/download/v2.3.0-nightly/${IPEXLLM_BUNDLE} && \
    tar xf ${IPEXLLM_BUNDLE} --strip-components=1 -C /

# ── Cleanup (same layer = actually reduces image size) ────────────────
RUN apt-get purge -y --auto-remove software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
           /var/log/*.log /var/log/apt/* /var/log/dpkg.log*

# ══════════════════════════════════════════════════════════════════════
# Runtime environment — Ollama server defaults
# See docs/intel-arc-a770-context-limits.md for VRAM / context guidance
# ══════════════════════════════════════════════════════════════════════

# Ollama server behaviour
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_NUM_PARALLEL=1
ENV OLLAMA_MAX_LOADED_MODELS=1
ENV OLLAMA_MAX_QUEUE=512
ENV OLLAMA_DEFAULT_KEEPALIVE=6h
ENV OLLAMA_KEEP_ALIVE=24h

# GPU layer offloading (999 = all layers on GPU)
ENV OLLAMA_NUM_GPU=999

# Intel IPEX-LLM / SYCL / Arc GPU tuning
# https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/fastchat_quickstart.md
ENV USE_XETLA=OFF
ENV ZES_ENABLE_SYSMAN=1
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV ENABLE_SDP_FUSION=1
ENV SYCL_CACHE_PERSISTENT=1

# NPU support (disabled by default; set to 1 if your CPU has an NPU)
# Arrow Lake (Series 2, 2xxK/2xxH):
# ENV IPEX_LLM_NPU_ARL=1
# Meteor Lake (Series 1, 1xxH):
# ENV IPEX_LLM_NPU_MTL=1

ENTRYPOINT ["/bin/bash", "/start-ollama.sh"]
