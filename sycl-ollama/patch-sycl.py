#!/usr/bin/env python3
"""
Patch upstream ggml-sycl to match ollama's modified ggml backend API.

As of ollama v0.16.1 (ggml commit ec98e200, llama.cpp tag b7437), the APIs
have converged and no patches are needed:
- graph_compute() already includes 'int batch_size' in both upstream and ollama
- GGML_TENSOR_FLAG_COMPUTE has been removed from both

For older ollama versions (e.g. v0.15.6, ggml commit a5bb8ba4), two patches
were required:
1. graph_compute() had an extra 'int batch_size' parameter (ollama addition)
2. GGML_TENSOR_FLAG_COMPUTE enum value was removed from ollama's ggml.h,
   so the skip-check in the compute loop had to be removed entirely
"""

import re
import sys

path = sys.argv[1]
with open(path, "r") as f:
    src = f.read()

original = src
applied = []

# 1. Fix graph_compute signature: add 'int batch_size' parameter
#    Only needed if the function does NOT already have batch_size
if "ggml_backend_sycl_graph_compute" in src and "int batch_size" not in src:
    src = re.sub(
        r'(static\s+(?:enum\s+)?ggml_status\s+ggml_backend_sycl_graph_compute\s*\([^)]*cgraph)\s*\)',
        r'\1, int batch_size)',
        src,
    )
    src = re.sub(
        r'(ggml_backend_sycl_graph_compute\([^)]*int\s+batch_size\)\s*\{)',
        r'\1\n    GGML_UNUSED(batch_size);',
        src,
    )
    if "int batch_size" in src:
        applied.append("batch_size parameter")

# 2. Remove GGML_TENSOR_FLAG_COMPUTE skip-check entirely.
#    Only needed if the flag is still referenced in the source.
if "GGML_TENSOR_FLAG_COMPUTE" in src:
    src = re.sub(
        r'\s*if\s*\(\(node->flags\s*&\s*GGML_TENSOR_FLAG_COMPUTE\)\s*==\s*0\)\s*\{\s*continue;\s*\}',
        '',
        src,
    )
    if "GGML_TENSOR_FLAG_COMPUTE" not in src:
        applied.append("GGML_TENSOR_FLAG_COMPUTE removed")

if src == original:
    print(f"No patches needed for {path} — APIs have converged")
    sys.exit(0)

with open(path, "w") as f:
    f.write(src)

for name in applied:
    print(f"  [OK] {name}")

print(f"Patched {path} successfully ({len(applied)} patches applied)")
