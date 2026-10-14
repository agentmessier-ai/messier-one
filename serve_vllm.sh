#!/bin/bash
# vLLM + this front end for one model directory.   ./serve_vllm.sh MODEL_DIR [front-end port, default 8080] [more server.py options]
# From v0.4 the model needs chunked prefill and prefix caching off; earlier revisions keep
# prefix caching.
set -eu
M=$1; PORT=${2:-8080}; VP=${VLLM_PORT:-8000}
if grep -q '"noncausal": *true' "$M/sys1_merge.json" 2>/dev/null; then
  EXTRA="--max-model-len 8192 --max-num-batched-tokens 8192 --no-enable-chunked-prefill --no-enable-prefix-caching"; FE="--max-state 7500 --max-total 8150"
else
  EXTRA="--max-model-len 10240 --enable-prefix-caching"; FE=""
fi
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$M" --served-model-name m --port $VP --max-num-seqs 32 --max-logprobs 256 \
  --logprobs-mode processed_logprobs --language-model-only $EXTRA &
until curl -s -m 3 localhost:$VP/health > /dev/null; do sleep 5; done
exec python server.py --model-dir "$M" --vllm http://127.0.0.1:$VP --served m --port "$PORT" $FE "${@:3}"
