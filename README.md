# Messier One front end

A TypeSafe-compatible `/v1/systemone` endpoint for Messier One. The model answers `choice`, `score` and `noul` questions with
probabilities read from one forward pass; no answer text is generated.

## Run

```bash
pip install -r requirements.txt   # vllm 0.30.0, transformers 5.18.0: the versions this was tested with
hf download agentmessier/messier-one --revision v0.1 --local-dir MODEL_DIR

# 1) the model (loopback only)
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve MODEL_DIR --served-model-name m --port 8000 \
  --max-model-len 10240 --max-num-seqs 32 --max-logprobs 256 --logprobs-mode processed_logprobs \
  --enable-prefix-caching --language-model-only
# 2) the /v1/systemone front end
python server.py --model-dir MODEL_DIR --vllm http://127.0.0.1:8000 --served m --port 8080
```

Wait until `GET http://127.0.0.1:8000/health` and `GET http://127.0.0.1:8080/health` return 200. After the download nothing
needs network access. One GPU with 24 GB is enough (weights about 10 GB, BF16).

## Request

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Order 8841: left earbud is dead, I want my money back. Order total 699."},
  "questions": {
    "route":  {"type": "choice", "instructions": "What should happen with this ticket?",
               "criteria": {"refund": "refund the order", "escalate": "send to a human agent", "ignore": null}},
    "urgent": {"type": "noul",   "instructions": "Does the customer sound urgent?"},
    "anger":  {"type": "score",  "instructions": "How upset is the customer?", "criteria": ["calm", "irritated", "furious"]}
  }}'
```

- `choice`: up to 255 options; `criteria` maps option name to a description (or `null`: the name is then the description).
- `score`: `criteria` is the ordered list of levels; the answer is the expected level.
- `noul`: the answer is P(true).
- `confidence` follows the TypeSafe definitions (choice: `(p_max - 1/n) / (1 - 1/n)`); `probabilities` are always returned.
- `--gate G` (off by default): a question whose top probability is below `G` is read again with the options reversed and
  the two readings are averaged. It costs about a third more tokens on typical traffic.
  Turn it on for questions with many similar options (dozens or more, e.g. moves in a game); on ordinary decisions it changes
  nothing measurable.

## Files

| file | what it is |
| --- | --- |
| `prompt.py` | the prompt text and label tokens the model was trained with |
| `wire.py` | option handling and the answer shape (probabilities, confidence) |
| `server.py` | the HTTP front end |
| `test_*.py` | `python -m unittest test_wire.py test_prompt.py test_server.py` (no GPU, no model needed) |

## Licence

Apache-2.0 (see `LICENSE`). This covers the code in this repository only. The model weights are CC BY-NC 4.0;
commercial use of the weights: agentmessier.ai@gmail.com.
