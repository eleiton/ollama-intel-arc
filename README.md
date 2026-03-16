# Run Ollama, Stable Diffusion and Automatic Speech Recognition with your Intel Arc GPU

[[Blog](https://blog.eleiton.dev/posts/llm-and-genai-in-docker/)]

Effortlessly deploy a Docker-based solution that uses [Open WebUI](https://github.com/open-webui/open-webui) as your user-friendly 
AI Interface and [Ollama](https://github.com/ollama/ollama) for integrating Large Language Models (LLM).

Additionally, you can run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [SD.Next](https://github.com/vladmandic/sdnext) docker containers to 
streamline Stable Diffusion capabilities.

You can also run an optional docker container with [OpenAI Whisper](https://github.com/openai/whisper) to perform Automatic Speech Recognition (ASR) tasks.

All these containers have been optimized for Intel Arc Series GPUs on Linux systems by using [Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

![screenshot](resources/open-webui.png)

## Tested Hardware

| Intel GPU | Status |
|---|---|
| Core Ultra 7 155H integrated Arc (Meteor Lake) | Verified |
| Arc A-series (A770, A750, A380) | Expected compatible |
| Data Center Flex / Max | Expected compatible |

## Documentation

* **[SYCL vs Vulkan — GPU Backend Comparison](docs/sycl-vs-vulkan.md)** — performance benchmarks (SYCL is 40–100% faster), three backend options (IPEX-LLM bundle, SYCL from source, upstream Vulkan), how `patch-sycl.py` works, and troubleshooting.
* **[Intel Arc A770 Context Length & VRAM Guide](docs/intel-arc-a770-context-limits.md)** — how to choose context length, KV cache quantization, and model size for 16 GB Intel Arc GPUs. Includes VRAM budget tables, per-model recommendations, and environment variable reference.
* **[Custom IPEX-LLM Dockerfile](ipex-ollama/Dockerfile)** — build your own Ollama image from scratch with pinned Intel GPU runtimes (Level Zero, IGC, compute-runtime) and the IPEX-LLM portable bundle. Uses BuildKit cache mounts for fast rebuilds.
* **[SYCL Source Build Dockerfile](sycl-ollama/Dockerfile)** — multi-stage build that compiles `ggml-sycl` from source with Intel oneAPI, paired with the official Ollama v0.16.1 binary. Includes [`patch-sycl.py`](sycl-ollama/patch-sycl.py) for backward-compatible API patching (no patches needed as of v0.16.1).
* **[docker-compose.yml](docker-compose.yml)** — fully documented Compose file with env-var driven configuration. All Intel GPU tuning knobs (SYCL, XeTLA, SDP fusion, KV cache, flash attention) are configurable via `${VAR:-default}` syntax and a `.env` file.

## Services
1. Ollama  
   * Runs llama.cpp and Ollama with IPEX-LLM on your Linux computer with Intel Arc GPU.  
   * Built following the guidelines from [Intel](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/DockerGuides/README.md).  
   * Uses the official [Intel ipex-llm docker image](https://hub.docker.com/r/intelanalytics/ipex-llm-inference-cpp-xpu) as the base container.
   * Uses the latest versions of required packages, prioritizing cutting-edge features over stability.  
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
Run the following commands to start your Ollama instance with Open WebUI
```bash
$ git clone https://github.com/eleiton/ollama-intel-arc.git
$ cd ollama-intel-arc
$ podman compose up
```

Alternatively, to use the **SYCL-from-source** build (newer Ollama, faster inference — see [SYCL vs Vulkan](docs/sycl-vs-vulkan.md)):
```bash
$ podman compose -f docker-compose.sycl-ollama.yml up --build
```

Additionally, if you want to run one or more of the image generation tools, run these command in a different terminal:

For ComfyUI
```bash
$ podman compose -f docker-compose.comfyui.yml up
```

For SD.Next
```bash
$ podman compose -f docker-compose.sdnext.yml up
```

If you want to run Whisper for automatic speech recognition, run this command in a different terminal:
```bash
$ podman compose -f docker-compose.whisper.yml up
```

## Validate
Run the following command to verify your Ollama instance is up and running
```bash
$ curl http://localhost:11434/
Ollama is running
```
When using Open WebUI, you should see this partial output in your console, indicating your arc gpu was detected
```bash
[ollama-intel-arc] | Found 1 SYCL devices:
[ollama-intel-arc] | |  |                   |                                       |       |Max    |        |Max  |Global |                     |
[ollama-intel-arc] | |  |                   |                                       |       |compute|Max work|sub  |mem    |                     |
[ollama-intel-arc] | |ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
[ollama-intel-arc] | |--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
[ollama-intel-arc] | | 0| [level_zero:gpu:0]|                     Intel Arc Graphics|  12.71|    128|    1024|   32| 62400M|         1.6.32224+14|
```

For the **SYCL-from-source** build (`docker-compose.sycl-ollama.yml`), you should see:
```bash
[sycl-ollama] | Listening on [::]:11434 (version 0.16.1)
[sycl-ollama] | inference compute  id="" library="" name=SYCL0 description="Intel(R) Arc(TM) Graphics" type=discrete total="28.0 GiB"
```

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
If there are new updates in the [ipex-llm-inference-cpp-xpu](https://hub.docker.com/r/intelanalytics/ipex-llm-inference-cpp-xpu) docker Image or in the Open WebUI docker Image, you may want to update your containers, to stay up to date.

Before any updates, be sure to stop your containers
```bash
$ podman compose down 
```

Then just run a pull command to retrieve the `latest` images.
```bash
$ podman compose pull
```


After that, you can run compose up to start your services again.
```bash
$ podman compose up
```

## Manually connecting to your Ollama container
You can connect directly to your Ollama container by running these commands:

```bash
$ podman exec -it ollama-intel-arc /bin/bash
$ /llm/ollama/ollama -v
```

## Project Structure

```
.
├── docker-compose.yml                # Main stack: Ollama (IPEX-LLM) + Open WebUI
├── docker-compose.sycl-ollama.yml    # SYCL-from-source Ollama + Open WebUI (alternative)
├── docker-compose.comfyui.yml        # ComfyUI image generation
├── docker-compose.sdnext.yml         # SD.Next image generation
├── docker-compose.whisper.yml        # OpenAI Whisper speech recognition
├── docker-compose.ramalama.yml       # RamaLama support
│
├── ipex-ollama/
│   └── Dockerfile                    # IPEX-LLM bundle build (Ollama v0.9.3, SYCL)
│
├── sycl-ollama/                      # SYCL-from-source build (Ollama v0.16.1)
│   ├── Dockerfile                    # Multi-stage: oneAPI build → minimal runtime
│   ├── patch-sycl.py                 # API compat patches (no-op since v0.16.1)
│   ├── start-ollama.sh               # Legacy entrypoint (from IPEX-LLM era)
│   └── test-glm-ocr.sh              # Vision model test script (glm-ocr)
│
├── comfyui/
│   └── Dockerfile                    # ComfyUI with Intel Extension for PyTorch
├── sdnext/
│   └── Dockerfile                    # SD.Next with Intel Extension for PyTorch
├── whisper/
│   └── Dockerfile                    # OpenAI Whisper with Intel Extension for PyTorch
├── ramalama/
│   └── Dockerfile                    # RamaLama container
│
├── docs/
│   ├── sycl-vs-vulkan.md             # SYCL vs Vulkan backend comparison
│   └── intel-arc-a770-context-limits.md  # VRAM & context length guide
│
└── resources/                        # Screenshots for README
```

## My development environment:
* Core Ultra 7 155H
* Intel® Arc™ Graphics (Meteor Lake-P)
* Fedora 41

## References 
* [Open WebUI documentation](https://docs.openwebui.com/)
* [Docker - Intel ipex-llm tags](https://hub.docker.com/r/intelanalytics/ipex-llm-serving-xpu/tags)
* [Docker - Intel extension for pytorch](https://hub.docker.com/r/intel/intel-extension-for-pytorch/tags)
* [GitHub - Intel ipex-llm tags](https://github.com/intel/ipex-llm/tags)
* [GitHub - Intel extension for pytorch](https://github.com/intel/intel-extension-for-pytorch/tags)