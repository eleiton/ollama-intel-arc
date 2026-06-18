# ollama-sycl

A Dockerfile that bolts a SYCL backend onto the official Ollama binary so it
can run on Intel GPUs (Arc discrete, Arc iGPU / Meteor Lake, Data Center
Max) via Intel oneAPI and Level Zero.

## Why this image exists

The official `ollama/ollama` Linux release ships these runners: **CUDA,
Vulkan, CPU** (plus ROCm as a separate download). There is **no SYCL runner**
in the prebuilt binary.

For Intel GPUs that means two practical options:

| Path | Pros | Cons |
| --- | --- | --- |
| Stock image + `OLLAMA_VULKAN=1` | Zero build; works today | Vulkan backend isn't as tuned for Intel GPUs as SYCL; Mesa driver bugs (e.g. fp16 on some iGPUs) |
| **This image (SYCL)** | Native Intel path via Level Zero + oneMKL/oneDNN; generally best on Intel **discrete** GPUs (benchmark vs Vulkan on iGPUs like Meteor Lake, where Vulkan is competitive) | Requires building from source with Intel's `icx`/`icpx` compilers (~6 GB build image) |

Upstream Ollama doesn't ship a precompiled `libggml-sycl.so` because building
it requires Intel's proprietary oneAPI compilers. So we build only that one
shared library locally and drop it next to the official Ollama binary.

## What it builds

A two-stage image:

### Stage 1 — `sycl-builder` (Intel oneAPI basekit, ~6 GB)

1. `git clone` Ollama at the version pinned in `OLLAMA_VERSION`.
2. Run cmake against `llama/server`.
   `FetchContent` pulls the pinned llama.cpp commit and applies Ollama's
   compat patch automatically.
3. Build *only* the `ggml-sycl` cmake target with `icx`/`icpx`. The result
   is `libggml-sycl.so`, a dynamically loadable backend module
   (`GGML_BACKEND_DL=ON`).
4. Collect the SYCL backend plus every oneAPI runtime `.so` it dlopens
   (SYCL/DPC++ runtime, Unified Runtime + Level Zero adapter, oneMKL,
   oneDNN, TBB, Intel compiler runtime, SPIR-V fallback kernels) into
   `/sycl-runner/`. Strip debug symbols.

### Stage 2 — runtime (`ubuntu:24.04`, ~2 GB)
1. Install Intel GPU userspace via `.deb`s pinned in `ARG`s:
   - **Level Zero** loader — required by the SYCL runtime to talk to the GPU.
   - **IGC** (Intel Graphics Compiler) — JIT-compiles SPIR-V kernels into
     GPU ISA at runtime.
   - **compute-runtime** + **GMM** — the Intel OpenCL/Level Zero driver
     proper. This is what actually executes work on the GPU.
2. Download the official Ollama Linux tarball at `OLLAMA_VERSION` and
   extract to `/usr`. Delete `cuda_*` and `vulkan` runners (not needed on
   Intel; saves ~1 GB).
3. Copy `/sycl-runner/` from stage 1 into `/usr/lib/ollama/sycl/`. Ollama
   auto-discovers backends by directory.

## Key cmake flags and why

```
-DCMAKE_C_COMPILER=icx
-DCMAKE_CXX_COMPILER=icpx     # SYCL requires Intel's compilers — gcc/clang won't do
-DBUILD_SHARED_LIBS=ON
-DGGML_BACKEND_DL=ON          # produce a dlopen-able plugin, not a static lib
-DGGML_NATIVE=OFF             # don't tune for the build host's CPU — keep portable
-DGGML_OPENMP=OFF             # avoid OpenMP — clashes with oneAPI's libiomp5
-DGGML_SYCL=ON
-DGGML_SYCL_TARGET=INTEL      # build kernels for Intel GPU (not NVIDIA/AMD via oneAPI)
-DGGML_SYCL_F16=ON            # compile fp16 math support in; toggle at runtime via env
```

## Pinned versions and how to bump them

| `ARG` | What | When to bump |
| --- | --- | --- |
| `OLLAMA_VERSION` | Ollama release tag | New Ollama release. Verify the pinned llama.cpp commit still builds `libggml-sycl.so` correctly and that flash attention + q4_0 KV cache produce correct output. |
| `COMPUTE_RUNTIME_VERSION` | Intel GPU driver (`intel/compute-runtime`) | Newer GPU support or perf fixes. |
| `LEVEL_ZERO_VERSION` | Level Zero loader | Usually bump with compute-runtime. |
| `IGC_VERSION` / `IGC_BUILD` | Intel Graphics Compiler | Usually bump with compute-runtime. |
| `GMM_VERSION` | Intel GPU memory manager | Usually bump with compute-runtime. |

oneAPI version is pinned via the `intel/oneapi-basekit:2025.2.2-…` tag in
stage 1's `FROM`.

## Build and run

Built and launched via the repo-root compose file:

```sh
docker compose -f docker-compose.ollama-sycl.yml up -d --build
```

This produces the image `ollama-sycl:local` and runs it with `/dev/dri`
mounted. Runtime tuning (context length, KV cache type, fp16, flash
attention, etc.) is in `../.env.example` — copy to `.env` and adjust.

## Known issues

- **oneAPI 2025.3.x causes a segfault during warmup** on Xe-LPG (Meteor Lake).
  Stage 1 is pinned to `intel/oneapi-basekit:2025.2.2`; do not upgrade until
  the regression is resolved upstream.
