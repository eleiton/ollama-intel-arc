ARG OLLAMA_VERSION=0.32.2
ARG COMPUTE_RUNTIME_VERSION=26.27.39122.11
ARG LEVEL_ZERO_VERSION=1.32.0
ARG IGC_VERSION=2.38.2
ARG IGC_BUILD=22051
ARG GMM_VERSION=22.10.0

# =============================================================================
# Stage 1: Build ggml-sycl backend from Ollama source using Intel oneAPI
# =============================================================================
FROM intel/oneapi-basekit:2025.2.2-0-devel-ubuntu24.04 AS sycl-builder

ARG OLLAMA_VERSION

# Clone Ollama at the target version.
RUN git clone --depth 1 --branch v${OLLAMA_VERSION} \
  https://github.com/ollama/ollama.git /ollama

WORKDIR /ollama

# Build the SYCL backend via the llama/server cmake subproject.
# FetchContent downloads the pinned llama.cpp commit (LLAMA_CPP_VERSION) and
# applies the Ollama compat patch automatically.
# After the build, find the MODULE (libggml-sycl.so) and copy it to /sycl-runner.
RUN cmake -S llama/server -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_F16=ON && \
  cmake --build build --parallel $(nproc) --target ggml-sycl && \
  mkdir -p /sycl-runner && \
  find /ollama/build -name "libggml-sycl.so" -exec cp {} /sycl-runner/ \;

# Collect the SYCL runner and its oneAPI runtime dependencies into /sycl-runner
RUN \
  # SYCL / DPC++ runtime \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl.so* /sycl-runner/ && \
  # Unified Runtime (oneAPI 2025+)  search multiple possible locations \
  find /opt/intel/oneapi -name 'libur_loader.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -name 'libur_adapter_level_zero.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  find /opt/intel/oneapi -maxdepth 4 -name 'libumf.so*' | head -3 | xargs -I{} cp {} /sycl-runner/ && \
  # oneDNN (enabled by GGML_SYCL_DNN=ON default) \
  cp /opt/intel/oneapi/dnnl/latest/lib/libdnnl.so* /sycl-runner/ 2>/dev/null; \
  # oneMKL \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/mkl/latest/lib/libmkl_tbb_thread.so* /sycl-runner/ && \
  # TBB \
  cp /opt/intel/oneapi/tbb/latest/lib/intel64/gcc*/libtbb.so* /sycl-runner/ && \
  # Intel compiler runtime \
  cp /opt/intel/oneapi/compiler/latest/lib/libsvml.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libimf.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libintlc.so* /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libirng.so /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libiomp5.so /sycl-runner/ && \
  # Level-zero PI plugin (legacy, may not exist in newer oneAPI) \
  cp /opt/intel/oneapi/compiler/latest/lib/libpi_level_zero.so* /sycl-runner/ 2>/dev/null; \
  # SYCL SPIR-V kernels (needed for bfloat16, complex math, etc.) \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-fallback*.spv /sycl-runner/ && \
  cp /opt/intel/oneapi/compiler/latest/lib/libsycl-native*.spv /sycl-runner/ && \
  # Strip debug symbols to reduce image size \
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

# Intel GPU runtimes: Level Zero, IGC, compute-runtime, GMM
RUN mkdir -p /tmp/gpu && cd /tmp/gpu && \
  wget https://github.com/oneapi-src/level-zero/releases/download/v${LEVEL_ZERO_VERSION}/libze1_${LEVEL_ZERO_VERSION}+u24.04_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${GMM_VERSION}_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}-0_amd64.deb && \
  dpkg -i *.deb && rm -rf /tmp/gpu

# Install official ollama binary + CPU runners (skip CUDA/Vulkan)
RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
  zstd -d | tar -xf - -C /usr && \
  rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/vulkan

# Install SYCL runner + oneAPI runtime libs from build stage
COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/

# Serve ollama on all interfaces
ENV OLLAMA_HOST=0.0.0.0

# Keep models loaded in memory
ENV OLLAMA_KEEP_ALIVE=24h

# Intel GPU tuning
ENV ZES_ENABLE_SYSMAN=1
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0

EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
