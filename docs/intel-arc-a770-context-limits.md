# Intel Arc A770 (16 GB) — Context Length & VRAM Guide

Running LLMs on an Intel Arc A770 (16 GB VRAM) requires balancing **model size**, **context length**, and **KV cache quantization**. This document explains the trade-offs and provides practical recommendations.

## How Ollama Chooses the Default Context Length

| GPU VRAM | Ollama default context |
| -------- | ---------------------- |
| < 24 GB  | **4 096 tokens**       |
| >= 24 GB | **8 192 tokens**       |

The Arc A770 has 16 GB, so Ollama defaults to only **4k context** — which is very short for most real-world tasks (RAG, coding agents, web search). That is why we override it with `OLLAMA_CONTEXT_LENGTH` in docker-compose.

## VRAM Budget Breakdown

VRAM is consumed by three main components during inference:

| Component        | Typical size (7B q4) | Scales with                        |
| ---------------- | -------------------- | ---------------------------------- |
| Model weights    | ~4 GB                | Model size & quantization          |
| KV cache         | 2–8 GB               | Context length x layers x KV quant |
| Runtime overhead | ~0.5–1 GB            | Fixed (driver, SYCL, allocator)    |

**Available for KV cache** ≈ 16 GB − model weights − overhead ≈ **10–11 GB** for a 7B q4 model.

## Context Length vs VRAM (approximate)

The following table shows approximate VRAM usage for the KV cache alone, for a typical 7B model with 32 layers:

| Context length | KV cache (f16) | KV cache (q8_0) | KV cache (q4_0) |
| -------------- | -------------- | --------------- | --------------- |
| 4 096          | ~1 GB          | ~0.5 GB         | ~0.25 GB        |
| 8 192          | ~2 GB          | ~1 GB           | ~0.5 GB         |
| 16 384         | ~4 GB          | ~2 GB           | ~1 GB           |
| 32 768         | ~8 GB          | ~4 GB           | ~2 GB           |
| 65 536         | ~16 GB (OOM)   | ~8 GB           | ~4 GB           |

## Recommended Settings by Model Size

### 7B models (Llama 3.1 8B, Mistral 7B, Qwen2.5 7B, Gemma2 9B)

| Goal            | Context | KV cache | Flash attn | Fits in 16 GB?      |
| --------------- | ------- | -------- | ---------- | ------------------- |
| Safe default    | 16 384  | q4_0     | yes        | Yes, ~5 GB free     |
| Quality balance | 16 384  | q8_0     | yes        | Yes, ~3 GB free     |
| Long context    | 32 768  | q4_0     | yes        | Yes, tight          |
| Maximum context | 65 536  | q4_0     | yes        | Borderline, may OOM |

### 13B models (Llama 2 13B, CodeLlama 13B)

| Goal            | Context | KV cache | Flash attn | Fits in 16 GB? |
| --------------- | ------- | -------- | ---------- | -------------- |
| Safe default    | 8 192   | q4_0     | yes        | Yes            |
| Quality balance | 8 192   | q8_0     | yes        | Yes, tight     |
| Long context    | 16 384  | q4_0     | yes        | Borderline     |

### 30B+ models

Generally **not recommended** on 16 GB. A q4 quantized 30B model consumes ~16 GB for weights alone, leaving nothing for KV cache. Consider:

- Offloading some layers to CPU with `OLLAMA_NUM_GPU` < 999
- Using very short context (2 048–4 096)
- Smaller models with longer context are usually better in practice

## Environment Variables Reference

These variables can be set in `docker-compose.yml` or as environment variables when running the container:

### Context & Memory

| Variable                 | Default | Description                                                                            |
| ------------------------ | ------- | -------------------------------------------------------------------------------------- |
| `OLLAMA_CONTEXT_LENGTH`  | `16384` | Token context window size. KV cache scales linearly with this.                         |
| `OLLAMA_KV_CACHE_TYPE`   | `q4_0`  | KV cache quantization: `f16`, `q8_0`, or `q4_0`. Lower = less VRAM, more quality loss. |
| `OLLAMA_FLASH_ATTENTION` | `1`     | Enable flash attention. 10–30% faster, lower peak memory.                              |
| `OLLAMA_NUM_GPU`         | `999`   | Layers offloaded to GPU. `999` = all. Reduce if model does not fit.                    |

### Intel Arc GPU Tuning

| Variable                                        | Default | Description                                                                   |
| ----------------------------------------------- | ------- | ----------------------------------------------------------------------------- |
| `USE_XETLA`                                     | `OFF`   | `OFF` = enable XeTLA tensor kernels (better perf). `ON` = disable (fallback). |
| `ZES_ENABLE_SYSMAN`                             | `0`     | `1` = Intel Sysman for GPU monitoring & power control.                        |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | `1`     | Immediate command lists — often reduces latency on Arc.                       |
| `ENABLE_SDP_FUSION`                             | `1`     | Fused scaled dot-product attention kernels.                                   |
| `SYCL_CACHE_PERSISTENT`                         | `1`     | Persistent SYCL kernel cache — faster second+ runs.                           |

### NPU (Neural Processing Unit)

| Variable           | Default | Description                                              |
| ------------------ | ------- | -------------------------------------------------------- |
| `IPEX_LLM_NPU_ARL` | `0`     | Set `1` for Arrow Lake (Core Ultra Series 2, 2xxK/2xxH). |
| `IPEX_LLM_NPU_MTL` | `0`     | Set `1` for Meteor Lake (Core Ultra Series 1, 1xxH).     |

## Tips

1. **Start conservative** — use `q4_0` KV cache + 16k context, then increase if you have headroom.
2. **Monitor VRAM** — run `intel_gpu_top` or `xpu-smi` on the host to see real-time usage.
3. **Flash attention is free performance** — always keep `OLLAMA_FLASH_ATTENTION=1`.
4. **SYCL persistent cache** saves significant startup time on repeat loads. Only disable if you see SYCL compilation errors.
5. **Multiple parallel requests** (`OLLAMA_NUM_PARALLEL > 1`) roughly multiply KV cache usage. On 16 GB, keep it at 1 unless using very short context or small models.
6. **Increase shared memory** — Docker defaults `/dev/shm` to only 64 MB. The SYCL/Level Zero runtime uses shared memory for kernel compilation caches and scratch buffers, and Ollama may memory-map model files through it. Set `shm_size: "16G"` in `docker-compose.yml` (or `--shm-size=16g` with `docker run`). This does **not** pre-allocate memory — it only sets the upper limit. Without it, large models can fail with `SIGBUS` or silent inference errors.

## Building a Custom Image

Instead of using the upstream `intelanalytics/ipex-llm-inference-cpp-xpu` image, you can build a custom image with pinned Intel GPU runtime versions using the [`ipex-ollama/Dockerfile`](../ipex-ollama/Dockerfile).

The Dockerfile uses BuildKit cache mounts for fast rebuilds and `ARG` version pins at the top:

| Component | ARG | Current version |
|-----------|-----|-----------------|
| Level Zero | `LEVEL_ZERO_VERSION` | 1.28.0 |
| Intel Graphics Compiler | `IGC_VERSION` / `IGC_BUILD` | 2.28.4 / 20760 |
| Compute Runtime | `COMPUTE_RUNTIME_VERSION` | 26.05.37020.3 |
| GMM Library | `GMMLIB_VERSION` | 22.9.0 |
| IPEX-LLM Ollama bundle | `IPEXLLM_BUNDLE` | ollama-ipex-llm-2.3.0b20250725 (Ollama v0.9.3) |

To build:

```bash
docker build -t ipex-ollama:latest ./ipex-ollama/
```

Then uncomment the `build:` section in [`docker-compose.yml`](../docker-compose.yml) and comment out the `image:` line to use your custom build.

## Configuring via docker-compose

All environment variables in [`docker-compose.yml`](../docker-compose.yml) use `${VAR:-default}` syntax, so you can override them by creating a `.env` file in the project root:

```env
OLLAMA_CONTEXT_LENGTH=32768
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_FLASH_ATTENTION=1
OLLAMA_NUM_GPU=999
```

Or by exporting them before running `docker compose up`.

## Troubleshooting

**OOM / model too large** — reduce `OLLAMA_CONTEXT_LENGTH`, switch `OLLAMA_KV_CACHE_TYPE` to `q4_0`, or use a smaller model quantization (Q4_0, Q4_K_M).

**Slow first inference** — SYCL JIT-compiles GPU kernels on first run. Keep `SYCL_CACHE_PERSISTENT=1` so subsequent runs are fast.

**`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`** — known Level-Zero regression on kernel 6.18+. See [SYCL vs Vulkan](sycl-vs-vulkan.md) for workarounds.

**Garbage output / wrong results** — may indicate an ABI mismatch if using a custom SYCL build. See [SYCL vs Vulkan — Troubleshooting](sycl-vs-vulkan.md#troubleshooting).

## Further Reading

- [SYCL vs Vulkan — GPU Backend Comparison](sycl-vs-vulkan.md)
- [Ollama context length docs](https://docs.ollama.com/context-length)
- [Ollama FAQ — flash attention](https://docs.ollama.com/faq#how-can-i-enable-flash-attention)
- [IPEX-LLM Docker guide](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/DockerGuides/README.md)
- [Intel compute-runtime releases](https://github.com/intel/compute-runtime/releases)
- [Level Zero releases](https://github.com/oneapi-src/level-zero/releases)
- [Intel Graphics Compiler releases](https://github.com/intel/intel-graphics-compiler/releases)
- [IPEX-LLM Ollama portable bundles](https://github.com/ipex-llm/ipex-llm/releases/tag/v2.3.0-nightly)
