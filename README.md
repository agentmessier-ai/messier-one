# Messier One

A small open-weights **decision model**. Give it a state (any text or JSON) and typed questions; it returns a probability for
every option from one forward pass. It does not generate text.

- Weights: [agentmessier/messier-one](https://huggingface.co/agentmessier/messier-one) on Hugging Face (Qwen3.5-4B fine-tune, about 10 GB)
- This repository: the serving code, a TypeSafe-compatible `POST /v1/systemone` endpoint
- Benchmark request: [JevBench #194](https://github.com/fstandhartinger/jevbench/issues/194)

```json
{"state": {"ticket": "Order 8841: left earbud is dead, I want my money back. Order total 699."},
 "questions": {"route": {"type": "choice", "instructions": "What should happen with this ticket?",
                         "criteria": {"refund": "refund the order", "escalate": "send to a human agent", "ignore": null}}}}
```

```json
{"answers": {"route": {"type": "choice", "choice": "refund", "confidence": 0.27,
                       "probabilities": {"refund": 0.51, "escalate": 0.46, "ignore": 0.03}}}}
```

Your code acts when the model is sure and hands the rest to a person: here it is torn between a refund and a human agent, and
says so.

## What it is good at

- **One forward pass, no generated text.** Every answer is a probability for each option, read at one position. On one
  RTX 5090 a decision takes about 50 ms (short documents) to 95 ms (long ones), server on loopback.
- **Up to 255 options in one question.** Routing to one of many tools, picking a category from a long list, choosing a move
  among many. On our own many-option test it is right 97.6% of the time with 2-26 options, 91.1% with 27-100 and 89.4% with
  101-255.
- **It reads the goal you give it.** The same document judged under a different goal gets a different answer. On our own
  held-out test (166 project descriptions, the same yes/no question under the original goal, the opposite goal and a goal
  never seen in training) it agrees with the reference answers 86-90% of the time under all three.
- **Probabilities you can use as they are.** They come from the model's own distribution with one fitted temperature per
  question type. Nothing is pushed toward 0 or 1 afterwards.
- **A drop-in endpoint.** The TypeSafe wire format: `choice`, `score` and `noul` questions, several per request.
- **Small.** One 24 GB GPU is enough.

The numbers above are our own measurements on our own test sets, whose reference answers were produced by larger language
models, not by people.

## Results

Public JevBench items (231), measured by us with JevBench's own runner and its unchanged `typesafe` adapter: serial requests,
single read, this server on one RTX 5090 (vLLM 0.30.0, BF16), loopback.

| tier | correct | accuracy | ECE | latency p50 / p95 |
| --- | --- | --- | --- | --- |
| easy | 48 / 48 | 100.0% | 0.003 | 46 ms / 52 ms |
| standard + judge (`original`) | 72 / 72 | 100.0% | 0.059 | 51 ms / 57 ms |
| hard | 77 / 111 | 69.4% | 0.092 | 95 ms / 171 ms |
| all | 197 / 231 | 85.3% | | |

These are our own measurements on the public items, not leaderboard results. Part of the training material was written for
this model in the families of JevBench's hard tier; no JevBench item, public or held out, was used for training, and every
training question was scanned against the public items before use.

## Run

```bash
git clone https://github.com/agentmessier-ai/messier-one && cd messier-one
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
needs network access.

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

What the released model answers:

```json
{"answers": {
  "route":  {"type": "choice", "choice": "refund", "confidence": 0.27,
             "probabilities": {"refund": 0.51, "escalate": 0.46, "ignore": 0.03}},
  "urgent": {"type": "noul", "noul": 0.11},
  "anger":  {"type": "score", "score": 0.59, "confidence": 0.41,
             "probabilities": {"0": 0.55, "1": 0.31, "2": 0.14}}}}
```

## The three question types

| type | you send | you get |
| --- | --- | --- |
| `choice` | `criteria`: option name -> description (or `null`: the name is then the description), or a list of names. 2 to 255 options | the likeliest option, a probability for each, `confidence` |
| `score` | `criteria`: the ordered list of levels, lowest first | the expected level as a number, a probability for each level, `confidence` |
| `noul` | only `instructions`: a yes/no question | `noul`: the probability that the answer is yes |

- `state` can be a string or any JSON value. Put there what the model should judge, and the goal or rules it should judge by.
- `instructions` is the question. Several questions in one request are answered independently.
- `confidence` follows the TypeSafe definitions (choice: `(p_max - 1/n) / (1 - 1/n)`): 0 at chance, 1 when certain.
  `probabilities` are always returned, so you can apply your own threshold.

## Option order, and reading twice (`--gate`)

**The problem.** The model reads the probability of each option's label (A, B, C, ...). It has a slight preference for some
positions, so the order in which you list the options can change the answer. On most questions it does not. It does when the
model is torn between two options, and when there are many options that look alike. Asking 300 questions once as written and
once with the options reversed, the answer changed on 6% of them.

**What does not pay.** Reading every question in several orders and averaging: four orders cost four times as much, helped
only on questions with many similar options, and changed nothing on ordinary decisions, where the model was already sure.

**What we do.** `python server.py ... --gate 0.5` reads a question once. Only if its top probability is below the threshold
is it read a second time with the options reversed, and the two readings are averaged. Off by default.

| test | single read | `--gate 0.5` |
| --- | --- | --- |
| 300 questions with 10-26 similar options (legal placements in a Tetris position) | 67.0% | 76.0% |
| many-option questions, 101-255 options | 89.4% | 92.9% |

Measured during development on the checkpoint just before the released one (same architecture, same readout). On ordinary
decisions the second reading changes nothing: the released model scores 85.3% on the public JevBench items with it and without.

- **Turn it on** for questions with many similar options: dozens of candidates, moves in a game, long category lists.
- **Leave it off** for ordinary decisions. It changes nothing there and costs tokens.
- **Cost.** About a third more tokens on typical traffic. A question that is read twice takes twice as long; on the Tetris
  questions above about half were.
- The threshold is compared with the top probability, not with the `confidence` field.
- Without `--gate` you can do the same check yourself: ask the question twice with the options in a different order. If the
  answer changes, treat it as "not sure".

## Limits

- Confidence can be off on unfamiliar domains.
- Arithmetic, date arithmetic and multi-step computation are unreliable: compute in code and put the result in the state.
- It answers the question it is asked from the state it is given. It has no memory between requests and takes no actions.

## Files

| file | what it is |
| --- | --- |
| `prompt.py` | the prompt text and label tokens the model was trained with |
| `wire.py` | option handling and the answer shape (probabilities, confidence) |
| `server.py` | the HTTP front end |
| `test_*.py` | `python -m unittest test_wire.py test_prompt.py test_server.py` (no GPU, no model needed) |

## Licence

- Code in this repository: Apache-2.0 (see `LICENSE`).
- Model weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), free for research and evaluation. For a
  commercial licence write to agentmessier.ai@gmail.com.
- The weights are a fine-tuned version of `Qwen/Qwen3.5-4B` (Apache-2.0).
