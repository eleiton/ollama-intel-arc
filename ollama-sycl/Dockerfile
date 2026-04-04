ARG OLLAMA_VERSION=0.20.2
ARG GGML_COMMIT=15bff84bf56651d6f991f166a2bf0f362996f7f9
ARG COMPUTE_RUNTIME_VERSION=26.09.37435.1
ARG LEVEL_ZERO_VERSION=1.28.0
ARG IGC_VERSION=2.30.1
ARG IGC_BUILD=20950
ARG GMM_VERSION=22.9.0

# =============================================================================
# Stage 1: Build ggml-sycl backend from ollama's ggml source using Intel oneAPI
# =============================================================================
FROM intel/oneapi-basekit:2025.2.2-0-devel-ubuntu24.04 AS sycl-builder

ARG OLLAMA_VERSION
ARG GGML_COMMIT

# Surgically replace Ollama's vendored ggml-sycl with a newer upstream version
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

# Apply API compatibility patches
COPY patch-sycl.py /tmp/patch-sycl.py
RUN python3 /tmp/patch-sycl.py ml/backend/ggml/ggml/src/ggml-sycl/ggml-sycl.cpp

# Build the SYCL backend as a dynamic library
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
  # SYCL SPIR-V kernels (needed for bfloat16, complex math, etc.)
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-fallback*.spv /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-native*.spv /sycl-runner/ && \
  # Strip debug symbols to reduce size
  strip --strip-unneeded /sycl-runner/*.so* 2>/dev/null; true

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

ARG OLLAMA_VERSION
ARG COMPUTE_RUNTIME_VERSION
ARG LEVEL_ZERO_VERSION
ARG IGC_VERSION
ARG IGC_BUILD
ARG GMM_VERSION

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
    wget https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/level-zero_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMM_VERSION}_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/gpu

# Install official ollama binary + CPU runners (skip CUDA/MLX/Vulkan)
RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
  zstd -d | tar -xf - -C /usr && \
  rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/mlx_* /usr/lib/ollama/vulkan

# Install SYCL runner from build stage
COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/

# Serve ollama on all interfaces
ENV OLLAMA_HOST=0.0.0.0

# Keep models loaded in memory
ENV OLLAMA_KEEP_ALIVE=24h

# Flash attention not supported by SYCL backend
ENV OLLAMA_FLASH_ATTENTION=false

# Intel GPU tuning
ENV ZES_ENABLE_SYSMAN=1
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0

# For Intel Core Ultra Processors (Series 1), code name Meteor Lake
ENV IPEX_LLM_NPU_MTL=1

EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
