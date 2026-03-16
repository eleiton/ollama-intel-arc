#!/bin/bash
export OLLAMA_NUM_GPU=999
export no_proxy=localhost,127.0.0.1
export ZES_ENABLE_SYSMAN=1
# [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Use OLLAMA_HOST and OLLAMA_KEEP_ALIVE from environment (set via docker-compose),
# falling back to sensible defaults if not set.
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

# [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
# export ONEAPI_DEVICE_SELECTOR=level_zero:0
# If you have more than one dGPUs, according to your configuration you can use configuration like below, it will use the first and second card.
# export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"

./ollama serve
