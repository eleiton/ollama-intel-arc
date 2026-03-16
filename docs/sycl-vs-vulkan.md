# SYCL vs Vulkan — Intel GPU Backends for Ollama

Ollama supports two GPU backends for Intel Arc/iGPU acceleration. This document explains the differences and helps you choose.

## Performance Comparison

Both backends run on Intel GPUs, but SYCL has a significant speed advantage:

| Intel GPU | Vulkan (tok/s) | SYCL (tok/s) | Gain |
|---|---|---|---|
| MTL iGPU (Core Ultra 155H) | ~8–11 | **~16** | +45–100% |
| ARL-H iGPU (Arrow Lake) | ~10–12 | **~17** | +40–70% |
| Arc A770 (16 GB) | ~30–35 | **~55** | +57–83% |
| Flex 170 | ~30–35 | **~50** | +43–67% |
| Data Center Max 1550 | — | **~73** | — |

*Benchmarks: Llama 2 7B Q4_0, llama.cpp, community-reported.*

## What Makes SYCL Faster

- **oneMKL / oneDNN** — Intel's optimized math and neural network libraries, hand-tuned per architecture
- **Level-Zero** — direct GPU communication with lower overhead than Vulkan's abstraction layer
- **Intel-tuned kernels** — `MUL_MAT` operations hand-optimized for each architecture (Meteor Lake, Arrow Lake, Arc, Flex, PVC)

## When to Use Vulkan Instead

- **No build step required** — upstream Ollama ships Vulkan support out of the box (`OLLAMA_VULKAN=1`)
- **Cross-vendor** — same backend works on AMD, NVIDIA, and Intel
- **Smaller image** — no oneAPI runtime libraries needed (~2 GB smaller)
- **No custom build** — upstream Ollama ships Vulkan pre-built, no compilation step needed
- **Kernel 6.18+ compatibility** — Vulkan avoids the known Level-Zero regression on kernel 6.18+

## Backend Options in This Repo

### Option 1: IPEX-LLM SYCL Bundle (current default)

Uses [`ipex-ollama/Dockerfile`](../ipex-ollama/Dockerfile) with the IPEX-LLM portable bundle.

| Attribute | Value |
|---|---|
| Ollama version | v0.9.3 |
| Backend | SYCL (IPEX-LLM patched llama.cpp) |
| Build time | ~2 min (download only) |
| Image size | **1.03 GB** |
| Status | Archived (Jan 2026), no further updates |

### Option 2: SYCL from Source (advanced)

Builds `ggml-sycl` from the exact llama.cpp commit that Ollama vendors, using Intel oneAPI. This unlocks a much newer Ollama version while keeping SYCL performance.

| Attribute | Value |
|---|---|
| Ollama version | v0.16.1+ |
| Backend | SYCL (ggml-sycl built from source) |
| Build time | ~90 s (compiles C++ with icpx) |
| Image size | **1.27 GB** |
| Status | Actively maintainable |

#### How the Source Build Works

Ollama ships the `ggml-sycl.h` header but intentionally excludes the SYCL implementation from its vendored ggml. The source build fills that gap. See [`sycl-ollama/Dockerfile`](../sycl-ollama/Dockerfile) for the full multi-stage build.

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Build  (intel/oneapi-basekit:2025.1.1)        │
│                                                         │
│  ollama v0.16.1 source ─┐                               │
│                         ├── cmake + icpx ── libggml-sycl.so
│  ggml-sycl @ ec98e200 ──┘                               │
│        ▲                                                │
│        └── patch-sycl.py (no-op since v0.16.1)          │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Runtime  (ubuntu:24.04)                       │
│                                                         │
│  ollama binary (official v0.16.1)                       │
│  + libggml-sycl.so + oneAPI runtime libs                │
│  + Intel GPU drivers (Level-Zero, IGC, compute-runtime) │
└─────────────────────────────────────────────────────────┘
```

**Stage 1 — Build** (from [`sycl-ollama/Dockerfile`](../sycl-ollama/Dockerfile)):

Clone Ollama and fetch the matching `ggml-sycl` source:

```dockerfile
FROM intel/oneapi-basekit:2025.1.1-0-devel-ubuntu24.04 AS sycl-builder

ARG OLLAMA_VERSION=0.16.1
ARG GGML_COMMIT=ec98e20021f7611db3bbcf6bb6629fed6e1ce4f0

RUN git clone --depth 1 --branch v${OLLAMA_VERSION} \
      https://github.com/ollama/ollama.git /ollama && \
    git init /tmp/llama.cpp && cd /tmp/llama.cpp && \
    git remote add origin https://github.com/ggml-org/llama.cpp.git && \
    git sparse-checkout set ggml/src/ggml-sycl && \
    git fetch --depth 1 origin ${GGML_COMMIT} && \
    git checkout FETCH_HEAD && \
    cp -r /tmp/llama.cpp/ggml/src/ggml-sycl \
      /ollama/ml/backend/ggml/ggml/src/ggml-sycl
```

Apply the API compatibility patches with [`patch-sycl.py`](../sycl-ollama/patch-sycl.py):

```dockerfile
COPY patch-sycl.py /tmp/patch-sycl.py
RUN python3 /tmp/patch-sycl.py ml/backend/ggml/ggml/src/ggml-sycl/ggml-sycl.cpp
```

As of v0.16.1, the upstream ggml and Ollama APIs have converged — **no patches are needed**. The script detects this and exits cleanly.

For older versions (e.g. v0.15.6), the patch fixed two divergences:

1. **`graph_compute` signature** — Ollama added an `int batch_size` parameter not present upstream. (Now in both.)
2. **`GGML_TENSOR_FLAG_COMPUTE` removal** — Ollama dropped this enum from `ggml.h`. Without the patch, the flag check caused all compute nodes to be skipped. (Now removed from both.)

Build the SYCL backend library:

```dockerfile
RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_SYCL=ON \
      -DGGML_SYCL_TARGET=INTEL \
      -DOLLAMA_RUNNER_DIR=sycl && \
    cmake --build build --parallel $(nproc) --target ggml-sycl
```

Collect runtime dependencies (oneAPI libs are ~800 MB in the SDK but only ~200 MB stripped):

```dockerfile
RUN mkdir -p /sycl-runner && \
    cp build/lib/ollama/libggml-sycl.so /sycl-runner/ && \
    cp /opt/intel/oneapi/compiler/latest/lib/libsycl.so* /sycl-runner/ && \
    cp /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so* /sycl-runner/ && \
    cp /opt/intel/oneapi/mkl/latest/lib/libmkl_intel_ilp64.so* /sycl-runner/ && \
    cp /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl_blas.so* /sycl-runner/ && \
    # ... TBB, oneDNN, Unified Runtime, compiler runtime ...
    strip --strip-unneeded /sycl-runner/*.so*
```

**Stage 2 — Runtime:**

```dockerfile
FROM ubuntu:24.04

# Install Intel GPU drivers (Level-Zero, IGC, compute-runtime)
# ... same as ipex-ollama/Dockerfile ...

# Install official ollama binary (skip CUDA/Vulkan runners)
ARG OLLAMA_VERSION=0.16.1
RUN wget -qO- "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | \
    zstd -d | tar -xf - -C /usr && \
    rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/mlx_* /usr/lib/ollama/vulkan

# Drop in the SYCL runner from Stage 1
COPY --from=sycl-builder /sycl-runner/ /usr/lib/ollama/sycl/

ENV OLLAMA_HOST=0.0.0.0:11434
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0

ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
```

The result is a standard Ollama install that transparently uses the SYCL backend for Intel GPUs. Ollama auto-discovers `libggml-sycl.so` in its runner directory.

#### Updating to a New Ollama Version

To track a new Ollama release:

1. Update `OLLAMA_VERSION` in `sycl-ollama/Dockerfile`
2. Find the ggml commit Ollama vendors: `git log --oneline ollama/ml/backend/ggml/ggml/` in the Ollama source
3. Update `GGML_COMMIT` to match
4. Rebuild — `patch-sycl.py` will auto-detect whether patches are needed and apply them if so (exits cleanly when no patches are required)

### Option 3: Upstream Ollama + Vulkan

Uses the official `ollama/ollama` Docker image with Vulkan enabled.

| Attribute | Value |
|---|---|
| Ollama version | v0.16.1 (latest) |
| Backend | Vulkan |
| Build time | None (pre-built image) |
| Image size | **5.63 GB** (includes CUDA runners) |
| Status | Actively maintained by Ollama team |

```bash
docker run -d \
  --device /dev/dri:/dev/dri \
  --shm-size 16G \
  -e OLLAMA_VULKAN=1 \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest
```

## Image Size Comparison

| Image | Size | Notes |
|---|---|---|
| **ipex-ollama** (this repo) | **1.03 GB** | IPEX-LLM bundle, smallest |
| **sycl-ollama** (this repo) | **1.27 GB** | SYCL from source + stripped oneAPI libs |
| `ollama/ollama:latest` | 5.63 GB | Upstream, includes CUDA/ROCm/Vulkan runners |
| `intelanalytics/ipex-llm-inference-cpp-xpu` | 19.9 GB | Intel's full base image |

Both custom images are **4–15x smaller** than alternatives because they include only the Intel GPU runtime libraries needed for SYCL inference, with no CUDA/ROCm/Vulkan overhead.

## Troubleshooting

**SYCL device not detected** — Ensure `/dev/dri` is accessible inside the container. Check logs for `SYCL0` in the device list. Verify Intel GPU drivers are installed on the host.

**"failed to sample token"** — Usually an ABI mismatch between ggml-sycl and Ollama's vendored ggml. The ggml commit used for building must match exactly what Ollama vendors.

**Model too large for VRAM** — Intel integrated GPUs share system memory. Increase `shm_size` in `docker-compose.yml` or use a smaller quantization (Q4_0, Q4_K_M). See the [VRAM guide](intel-arc-a770-context-limits.md).

**Slow first inference** — SYCL JIT-compiles GPU kernels on first run. Set `SYCL_CACHE_PERSISTENT=1` so compiled kernels are cached for subsequent runs.

**`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` on kernel 6.18+** — Known Level-Zero regression. Workarounds: downgrade kernel, or switch to Vulkan backend.

## Tested Hardware

| Intel GPU | Status |
|---|---|
| Core Ultra 7 155H integrated Arc (Meteor Lake) | Verified |
| Arc A-series (A770, A750, A380) | Expected compatible |
| Data Center Flex / Max | Expected compatible |

**Requirements:** Ubuntu 24.04+, Docker with Compose, Intel GPU with Level-Zero driver support.

## References

- [llama.cpp SYCL backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Intel oneAPI base toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
- [Intel GPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html)
- [Ollama Vulkan PR #11835](https://github.com/ollama/ollama/pull/11835)
- [IPEX-LLM archived repo](https://github.com/intel/ipex-llm)
