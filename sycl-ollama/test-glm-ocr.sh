#!/usr/bin/env bash
# Test glm-ocr:q8_0 vision model via ollama HTTP API
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL="${MODEL:-glm-ocr:q8_0}"
IMAGE="${1:-./image.png}"
PROMPT="${2:-Text Recognition:}"
#PROMPT="Table Recognition"
NUM_CTX="${NUM_CTX:-2048}"

TMPDIR="${TMPDIR:-/tmp}"
PAYLOAD_FILE=$(mktemp "$TMPDIR/ollama-req-XXXXXX.json")
RESPONSE_FILE=$(mktemp "$TMPDIR/ollama-resp-XXXXXX.json")
trap 'rm -f "$PAYLOAD_FILE" "$RESPONSE_FILE"' EXIT

if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Image not found: $IMAGE" >&2
  echo "Usage: $0 [image_path] [prompt]" >&2
  exit 1
fi

echo "--- Config ---"
echo "  URL:     $OLLAMA_URL"
echo "  Model:   $MODEL"
echo "  Image:   $IMAGE ($(wc -c <"$IMAGE") bytes)"
echo "  Prompt:  $PROMPT"
echo "  num_ctx: $NUM_CTX"
echo ""

# --- Check ollama is reachable ---
echo -n "Checking ollama... "
if ! curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "FAILED — ollama not reachable at $OLLAMA_URL" >&2
  exit 1
fi
echo "OK"

# --- Build JSON payload via python (handles large base64 safely) ---
echo -n "Building request... "
python3 <<PYEOF
import base64, json

with open("$IMAGE", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "$MODEL",
    "prompt": "$PROMPT",
    "images": [img_b64],
    "stream": False,
    "options": {"num_ctx": $NUM_CTX}
}

with open("$PAYLOAD_FILE", "w") as f:
    json.dump(payload, f)

print(f"OK ({len(img_b64)} chars base64)")
PYEOF

# --- Send request ---
echo ""
echo "--- Sending request (this may take a while on CPU) ---"
echo ""

START_SEC=$(date +%s)

HTTP_CODE=$(curl -s -o "$RESPONSE_FILE" -w "%{http_code}" \
  "$OLLAMA_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE" \
  --max-time 300)

END_SEC=$(date +%s)
ELAPSED=$((END_SEC - START_SEC))

echo "--- HTTP $HTTP_CODE (${ELAPSED}s) ---"
echo ""

if [[ "$HTTP_CODE" != "200" ]]; then
  echo "ERROR: Request failed with HTTP $HTTP_CODE" >&2
  echo ""
  cat "$RESPONSE_FILE" >&2
  echo "" >&2
  echo ""
  echo "--- Recent container logs ---"
  docker logs ollama-intel-gpu 2>&1 | tail -10
  exit 1
fi

# --- Parse and display response ---
python3 <<PYEOF
import json

with open("$RESPONSE_FILE") as f:
    data = json.load(f)

resp = data.get("response", "")
eval_count = data.get("eval_count", 0)
eval_dur = data.get("eval_duration", 0)
prompt_count = data.get("prompt_eval_count", 0)
prompt_dur = data.get("prompt_eval_duration", 0)

print(resp)
print()
print("--- Stats ---")
print(f"  Done:        {data.get('done', False)}")
print(f"  Tokens:      {eval_count}")
if eval_dur > 0 and eval_count > 0:
    print(f"  Speed:       {eval_count / (eval_dur / 1e9):.1f} tok/s")
if prompt_dur > 0 and prompt_count > 0:
    print(f"  Prompt eval: {prompt_count / (prompt_dur / 1e9):.1f} tok/s")
print(f"  Wall time:   {$ELAPSED}s")
PYEOF
