# Run Ollama, Stable Diffusion and Automatic Speech Recognition with your Intel Arc GPU

[[Blog](https://blog.eleiton.dev/posts/llm-and-genai-in-docker/)]

Effortlessly deploy a Docker-based solution that uses [Open WebUI](https://github.com/open-webui/open-webui) as your user-friendly 
AI Interface and [Ollama](https://github.com/ollama/ollama) for integrating Large Language Models (LLM).

Additionally, you can run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [SD.Next](https://github.com/vladmandic/sdnext) docker containers to 
streamline Stable Diffusion capabilities.

You can also run an optional docker container with [OpenAI Whisper](https://github.com/openai/whisper) to perform Automatic Speech Recognition (ASR) tasks.

The Ollama container runs a **native [SYCL](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) (or [Vulkan](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)) llama.cpp backend** built directly from upstream Ollama — no IPEX-LLM. The Stable Diffusion and Whisper containers are still optimized for Intel Arc GPUs using [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

![screenshot](resources/open-webui.png)

## Services
1. Ollama
   * Runs Ollama with a **native llama.cpp `ggml-sycl` backend**, compiled from Ollama source against Intel® oneAPI (`icpx` / Level Zero). No IPEX-LLM dependency.
   * The image is built locally in two stages (see [`ollama-sycl/Dockerfile`](ollama-sycl/Dockerfile)): stage 1 builds `libggml-sycl.so` with oneAPI; stage 2 drops it next to the official Ollama binary on a slim Ubuntu runtime with the Intel GPU user-space drivers (Level Zero, compute-runtime, IGC, GMM).
   * A **Vulkan** alternative is also provided ([`docker-compose.ollama-vulkan.yml`](docker-compose.ollama-vulkan.yml)) using the stock `ollama/ollama` image. On Meteor Lake / Xe-LPG iGPUs the Vulkan backend is often competitive with SYCL while requiring no custom build — worth benchmarking on your hardware.
   * Runtime behavior is tuned via a `.env` file (see [`.env.example`](.env.example)) — context length, KV-cache type, flash attention, GPU offload, etc.
   * Exposes port `11434` for connecting other tools to your Ollama service.

2. Open WebUI  
   * Uses the official distribution of Open WebUI.  
   * `WEBUI_AUTH` is turned off for authentication-free usage.  
   * `ENABLE_OPENAI_API` and `ENABLE_OLLAMA_API` flags are set to off and on, respectively, allowing interactions via Ollama only.
   * `ENABLE_IMAGE_GENERATION` is set to true, allowing you to generate images from the UI.
   * `IMAGE_GENERATION_ENGINE` is set to automatic1111 (SD.Next is compatible).

3. ComfyUI
   * The most powerful and modular diffusion model GUI, api and backend with a graph/nodes interface.
   * Uses as the base container the official [Intel® Extension for PyTorch](https://pytorch-extension.intel.com/installation?platform=gpu)

4. SD.Next
   * All-in-one for AI generative image based on Automatic1111
   * Uses as the base container the official [Intel® Extension for PyTorch](https://pytorch-extension.intel.com/installation?platform=gpu)
   * Uses a customized version of the SD.Next [docker file](https://github.com/vladmandic/sdnext/blob/dev/configs/Dockerfile.ipex), making it compatible with the Intel Extension for Pytorch image.

5. OpenAI Whisper
   * Robust Speech Recognition via Large-Scale Weak Supervision
   * Uses as the base container the official [Intel® Extension for PyTorch](https://pytorch-extension.intel.com/installation?platform=gpu)

## Setup

First, create your `.env` file from the example and adjust it to your hardware if needed:
```bash
$ git clone https://github.com/eleiton/ollama-intel-arc.git
$ cd ollama-intel-arc
$ cp .env.example .env
```

Then start Ollama + Open WebUI with the **native SYCL** backend (this builds the Ollama image locally the first time):
```bash
$ podman compose -f docker-compose.ollama-sycl.yml up -d --build
```

Alternatively, use the **Vulkan** backend (no local build — pulls the stock Ollama image):
```bash
$ podman compose -f docker-compose.ollama-vulkan.yml up -d
```

> The repository also ships a legacy `docker-compose.yml` based on the now-outdated `intelanalytics/ipex-llm-inference-cpp-xpu` image. It is kept for reference only; the native SYCL/Vulkan composes above are the recommended path.

Additionally, if you want to run one or more of the image generation tools, run these command in a different terminal:

For ComfyUI
```bash
$ podman compose -f docker-compose.comfyui.yml up -d
```

For SD.Next
```bash
$ podman compose -f docker-compose.sdnext.yml up -d
```

If you want to run Whisper for automatic speech recognition, run this command in a different terminal:
```bash
$ podman compose -f docker-compose.whisper.yml up -d
```

## Configuration

Ollama runtime behavior is controlled through environment variables in your `.env` file (passed through by the compose files). The defaults in [`.env.example`](.env.example) are tuned for an Intel Arc Graphics (Meteor Lake-P) integrated GPU with shared/UMA memory. Key settings:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Larger contexts grow the KV cache and reduce the model size that fits. |
| `OLLAMA_KV_CACHE_TYPE` | `q4_0` | Quantized KV cache saves memory. On a UMA iGPU with plenty of RAM, try `f16` or `q8_0` — it can be faster and higher quality at no real memory cost. |
| `OLLAMA_FLASH_ATTENTION` | `true` | Works on both the SYCL and Vulkan paths on this iGPU. |
| `OLLAMA_NUM_GPU` | `999` | Offload all transformer layers to the GPU. |
| `OLLAMA_NUM_PARALLEL` | `1` | One request at a time — UMA iGPUs are bandwidth-bound, so parallelism gives no throughput gain. |
| `OLLAMA_KEEP_ALIVE` | `2h` | Keep models resident to avoid reload latency. |
| `GGML_SYCL_F16` | `1` | Enable fp16 math in the SYCL backend. |

See [`.env.example`](.env.example) for the full annotated list.

## Validate
Run the following command to verify your Ollama instance is up and running
```bash
$ curl http://localhost:11434/
Ollama is running
```
When using Open WebUI, you should see this partial output in your console, indicating your arc gpu was detected
```bash
[ollama-sycl] | Found 1 SYCL devices:
[ollama-sycl] | |  |                   |                                       |       |Max    |        |Max  |Global |                     |
[ollama-sycl] | |  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
[ollama-sycl] | |ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
[ollama-sycl] | |--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
[ollama-sycl] | | 0| [level_zero:gpu:0]|                     Intel Arc Graphics|  12.71|    128|    1024|   32| 62400M|         1.6.32224+14|
```
(The Vulkan backend logs `ggml_vulkan: Found ... Intel(R) Arc(TM) Graphics` instead.)

## Using Image Generation
* Open your web browser to http://localhost:7860 to access the SD.Next web page.
* For the purposes of this demonstration, we'll use the [DreamShaper](https://civitai.com/models/4384/dreamshaper) model.
* Follow these steps:
* Download the  `dreamshaper_8` model by clicking on its image (1).
* Wait for it to download (~2GB in size) and then select it in the dropbox (2).
* (Optional) If you want to stay in the SD.Next UI, feel free to explore (3).
![screenshot](resources/sd.next.png)
* For more information on using SD.Next, refer to the official [documentation](https://vladmandic.github.io/sdnext-docs/).
* Open your web browser to http://localhost:4040 to access the Open WebUI web page.
* Go to the administrator [settings](http://localhost:4040/admin/settings) page.
* Go to the Image section (1)
* Make sure all settings look good, and validate them pressing the refresh button (2)
* (Optional) Save any changes if you made them. (3)
![screenshot](resources/open-webui-settings.png)
* For more information on using Open WebUI, refer to the official [documentation](https://docs.openwebui.com/)
* That's it, go back to Open WebUI main page and start chatting.  Make sure to select the `Image` button to indicate you want to generate Images.
![screenshot](resources/open-webui-chat.png)

## Using Automatic Speech Recognition
* This is an example of a command to transcribe audio files:
```bash
  podman exec -it  whisper-ipex whisper https://www.lightbulblanguages.co.uk/resources/ge-audio/hobbies-ge.mp3 --device xpu --model small --language German --task transcribe
```
* Response:
```bash
  [00:00.000 --> 00:08.000]  Ich habe viele Hobbys. In meiner Freizeit mache ich sehr gerne Sport, wie zum Beispiel Wasserball oder Radfahren.
  [00:08.000 --> 00:13.000]  Außerdem lese ich gerne und lerne auch gerne Fremdsprachen.
  [00:13.000 --> 00:19.000]  Ich gehe gerne ins Kino, höre gerne Musik und treffe mich mit meinen Freunden.
  [00:19.000 --> 00:22.000]  Früher habe ich auch viel Basketball gespielt.
  [00:22.000 --> 00:26.000]  Im Frühling und im Sommer werde ich viele Radtouren machen.
  [00:26.000 --> 00:29.000]  Außerdem werde ich viel schwimmen gehen.
  [00:29.000 --> 00:33.000]  Am liebsten würde ich das natürlich im Meer machen.
```
* This is an example of a command to translate audio files:
```bash
  podman exec -it  whisper-ipex whisper https://www.lightbulblanguages.co.uk/resources/ge-audio/hobbies-ge.mp3 --device xpu --model small --language German --task translate
```
* Response:
```bash
  [00:00.000 --> 00:02.000]  I have a lot of hobbies.
  [00:02.000 --> 00:05.000]  In my free time I like to do sports,
  [00:05.000 --> 00:08.000]  such as water ball or cycling.
  [00:08.000 --> 00:10.000]  Besides, I like to read
  [00:10.000 --> 00:13.000]  and also like to learn foreign languages.
  [00:13.000 --> 00:15.000]  I like to go to the cinema,
  [00:15.000 --> 00:16.000]  like to listen to music
  [00:16.000 --> 00:19.000]  and meet my friends.
  [00:19.000 --> 00:22.000]  I used to play a lot of basketball.
  [00:22.000 --> 00:26.000]  In spring and summer I will do a lot of cycling tours.
  [00:26.000 --> 00:29.000]  Besides, I will go swimming a lot.
  [00:29.000 --> 00:33.000]  Of course, I would prefer to do this in the sea.
```
* To use your own audio files instead of web files, place them in the `~/whisper-files` folder and access them like this:
```bash
  podman exec -it  whisper-ipex whisper YOUR_FILE_NAME.mp3 --device xpu --model small --task translate
```

## Updating the containers

For the **native SYCL** Ollama image, updates come from rebuilding against a newer Ollama release. Bump `OLLAMA_VERSION` (and, if needed, the Intel GPU driver pins) at the top of [`ollama-sycl/Dockerfile`](ollama-sycl/Dockerfile), then rebuild:
```bash
$ podman compose -f docker-compose.ollama-sycl.yml build --no-cache
$ podman compose -f docker-compose.ollama-sycl.yml up -d
```

For the **Vulkan** image, bump the `ollama/ollama` tag in [`docker-compose.ollama-vulkan.yml`](docker-compose.ollama-vulkan.yml) and pull:
```bash
$ podman compose -f docker-compose.ollama-vulkan.yml pull
$ podman compose -f docker-compose.ollama-vulkan.yml up -d
```

For Open WebUI and the other `latest`-tagged images, stop the stack and pull:
```bash
$ podman compose -f docker-compose.ollama-sycl.yml down
$ podman compose -f docker-compose.ollama-sycl.yml pull open-webui
$ podman compose -f docker-compose.ollama-sycl.yml up -d
```

## Manually connecting to your Ollama container
You can connect directly to your Ollama container by running these commands:

```bash
$ podman exec -it ollama-sycl /bin/bash
$ ollama -v
```
(Use `ollama-vulkan` as the container name if you are running the Vulkan compose.)

## My development environment:
* Core Ultra 7 155H
* Intel® Arc™ Graphics (Meteor Lake-P)
* Fedora 43

## References 
* [Open WebUI documentation](https://docs.openwebui.com/)
* [Ollama](https://github.com/ollama/ollama)
* [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
* [llama.cpp Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan)
* [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
* [Intel compute-runtime (Level Zero / OpenCL driver)](https://github.com/intel/compute-runtime)
* [Docker - Intel extension for pytorch](https://hub.docker.com/r/intel/intel-extension-for-pytorch/tags)
* [GitHub - Intel extension for pytorch](https://github.com/intel/intel-extension-for-pytorch/tags)
