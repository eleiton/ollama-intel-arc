# =============================================================================
# Stage 1: Build ggml-sycl backend from ollama's ggml source using Intel oneAPI
# =============================================================================
FROM intel/oneapi-basekit:2025.1.1-0-devel-ubuntu24.04 AS sycl-builder

ARG OLLAMA_VERSION=0.16.1

# Clone ollama source and the MATCHING ggml-sycl source from upstream llama.cpp.
# ollama v0.16.1 vendors ggml at commit ec98e200 (tag b7437) — we MUST use the
# same commit to ensure struct layouts, operation enums, and internal APIs match.
# (ollama excludes ggml-sycl from its vendored ggml, but keeps the header)
ARG GGML_COMMIT=ec98e20021f7611db3bbcf6bb6629fed6e1ce4f0
RUN git clone --depth 1 --branch v${OLLAMA_VERSION} \
  https://github.com/ollama/ollama.git /ollama && \
  git init /tmp/llama.cpp && \
  cd /tmp/llama.cpp && \
  git remote add origin https://github.com/ggml-org/llama.cpp.git && \
  git sparse-checkout set ggml/src/ggml-sycl && \
  git fetch --depth 1 origin ${GGML_COMMIT} && \
  git checkout FETCH_HEAD && \
  cp -r /tmp/llama.cpp/ggml/src/ggml-sycl \
  /ollama/ml/backend/ggml/ggml/src/ggml-sycl && \
  rm -rf /tmp/llama.cpp

WORKDIR /ollama

# Apply API compatibility patches if needed.
# As of v0.16.1, the upstream ggml-sycl and ollama's vendored ggml have converged:
# - graph_compute now includes 'int batch_size' in both
# - GGML_TENSOR_FLAG_COMPUTE has been removed from both
# The patch script exits gracefully (code 0) when no changes are needed.
COPY patch-sycl.py /tmp/patch-sycl.py
RUN python3 /tmp/patch-sycl.py ml/backend/ggml/ggml/src/ggml-sycl/ggml-sycl.cpp

# Build the SYCL backend as a dynamic library
# Note: oneAPI env is already set in the base image, no need to source setvars.sh
RUN cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DOLLAMA_RUNNER_DIR=sycl && \
  cmake --build build --parallel $(nproc) --target ggml-sycl

# Collect the SYCL runner and its oneAPI runtime dependencies into /sycl-runner
RUN mkdir -p /sycl-runner && \
  cp build/lib/ollama/libggml-sycl.so /sycl-runner/ && \
  # SYCL / DPC++ runtime
  for f in libsycl.so*; do true; done && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl.so* /sycl-runner/ && \
  # Unified Runtime (oneAPI 2025+) — search multiple possible locations
  find /opt/intel/oneapi -name 'libur_loader.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -name 'libur_adapter_level_zero.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -maxdepth 4 -name 'libumf.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  # oneDNN
  cp /opt/intel/oneapi/dnnl/latest/lib/libdnnl.so* /sycl-runner/ 2>/dev/null; \
  # oneMKL
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_tbb_thread.so* /sycl-runner/ && \
  # TBB
  cp /opt/intel/oneapi/tbb/latest/lib/intel64/gcc*/libtbb.so* /sycl-runner/ && \
  # Intel compiler runtime
  cp /opt/intel/oneapi/compiler/latest/lib/libsvml.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libimf.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libintlc.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libirng.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libiomp5.so /sycl-runner/ && \
  # Level-zero PI plugin (legacy, may not exist)
  cp /opt/intel/oneapi/compiler/latest/lib/libpi_level_zero.so* /sycl-runner/ 2>/dev/null; \
  # SYCL SPIR-V fallback kernels (needed for bfloat16, complex math, etc.)
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-fallback*.spv /sycl-runner/ && \
  # Strip debug symbols to reduce size
  strip --strip-unneeded /sycl-runner/*.so* 2>/dev/null; true

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
  TZ=America/Los_Angeles

# Base packages
RUN apt-get update && \
  apt-get install --no-install-recommends -q -y \
  ca-certificates \
  wget \
  zstd \
  ocl-icd-libopencl1 \
  libhwloc15 && \
  rm -rf /var/lib/apt/lists/*

# Intel GPU runtimes (release 26.05.37020.3)
# Provides level-zero, IGC, compute-runtime for Intel GPU kernel support
RUN mkdir -p /tmp/gpu && cd /tmp/gpu && \
  wget https://github.com/oneapi-src/level-zero/releases/download/v1.28.0/level-zero_1.28.0+u24.04_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.28.4/intel-igc-core-2_2.28.4+20760_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.28.4/intel-igc-opencl-2_2.28.4+20760_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.05.37020.3/intel-ocloc-dbgsym_26.05.37020.3-0_amd64.ddeb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.05.37020.3/intel-ocloc_26.05.37020.3-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.05.37020.3/intel-opencl-icd_26.05.37020.3-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.05.37020.3/libigdgmm12_22.9.0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.05.37020.3/libze-intel-gpu1_26.05.37020.3-0_amd64.deb && \
  dpkg -i *.deb *.ddeb && rm -rf /tmp/gpu

# Install official ollama binary + CPU runners (skip CUDA/MLX/Vulkan)
ARG OLLAMA_VERSION=0.16.1
RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
  zstd -d | tar -xf - -C /usr && \
  rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/mlx_* /usr/lib/ollama/vulkan

# Install SYCL runner from build stage
COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/

# Clean up
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  apt-get autoremove -y --purge 2>/dev/null; \
  apt-get autoclean -y 2>/dev/null; true

# Serve ollama on all interfaces
ENV OLLAMA_HOST=0.0.0.0:11434

# Keep models loaded in memory
ENV OLLAMA_KEEP_ALIVE=24h
ENV OLLAMA_DEFAULT_KEEPALIVE=6h

# Concurrency and resource limits
ENV OLLAMA_NUM_PARALLEL=1
ENV OLLAMA_MAX_LOADED_MODELS=1
ENV OLLAMA_MAX_QUEUE=512
ENV OLLAMA_MAX_VRAM=0

# Use all GPU layers
ENV OLLAMA_NUM_GPU=999

# Intel GPU tuning
ENV ZES_ENABLE_SYSMAN=1
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0

# For Intel Core Ultra Processors (Series 1), code name Meteor Lake
#ENV IPEX_LLM_NPU_MTL=1

EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
