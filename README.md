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
{"answers": {"route": {"type": "choice", "choice": "refund", "confidence": 0.38,
                       "probabilities": {"refund": 0.58, "escalate": 0.37, "ignore": 0.05}}}}
```

Your code acts when the model is sure and hands the rest to a person: here it is torn between a refund and a human agent, and
says so.

## What it is good at

- **One forward pass, no generated text.** Every answer is a probability for each option, read at one position. On one
  RTX 3090 a decision takes about 80 ms (short documents) to 145 ms (long ones), server on loopback.
- **Up to 255 options in one question.** Routing to one of many tools, picking a category from a long list, choosing a move
  among many. On our own many-option test it is right 97.9% of the time with 2-26 options, 93.8% with 27-100 and 92.9% with
  101-255 (single read).
- **It reads the goal you give it.** The same document judged under a different goal gets a different answer. On our own
  held-out test (166 project descriptions, the same yes/no question under three different goals) it agrees with the reference answers 90-92% of the time under all three.
- **Calibrated probabilities.** `choice` and `score` answers are returned as they are. A yes/no answer commits: see "Yes/no answers" below.
- **A drop-in endpoint.** The TypeSafe wire format: `choice`, `score` and `noul` questions, several per request.
- **Small.** One 24 GB GPU is enough.

The numbers above are our own measurements on our own test sets.

## Results

Public JevBench items (231), measured by us with JevBench's own runner and its unchanged `typesafe` adapter: serial requests,
this server started with `serve_vllm.sh` and its defaults (second reading below 0.5, see `--gate`) on one RTX 5090 (vLLM 0.30.0, BF16), loopback. Model
revision v0.4.1 (the weights of v0.4 with changed temperatures; v0.4 measured 206 / 231 on the same card).

| tier | correct | accuracy | ECE | latency p50 / p95 |
| --- | --- | --- | --- | --- |
| easy | 48 / 48 | 100.0% | 0.034 | 39 ms / 41 ms |
| standard + judge (`original`) | 72 / 72 | 100.0% | 0.075 | 38 ms / 40 ms |
| hard | 85 / 111 | 76.6% | 0.085 | 46 ms / 164 ms |
| all | 205 / 231 | 88.7% | | |

These are our own measurements on the public items, not leaderboard results. Part of the training material was written for
this model in the families of JevBench's hard tier; no JevBench item, public or held out, was used for training, and every
training question was scanned against the public items before use.

## Run

```bash
git clone https://github.com/agentmessier-ai/messier-one && cd messier-one
pip install -r requirements.txt   # vllm 0.30.0, transformers 5.18.0: the versions this was tested with
hf download agentmessier/messier-one --revision v0.4.1 --local-dir MODEL_DIR

# vLLM on 127.0.0.1:8000 (loopback only) and the /v1/systemone front end on port 8080
./serve_vllm.sh MODEL_DIR 8080
```

`serve_vllm.sh` starts vLLM with the settings the model needs and then the front end; extra arguments go to `server.py`
(for example `./serve_vllm.sh MODEL_DIR 8080 --gate 0`). `VLLM_PORT` changes vLLM's port.

Without vLLM, in one process (pure transformers; slower, no prefix cache, one 24 GB GPU):

```bash
python server.py --model-dir MODEL_DIR --engine transformers --port 8080
```

Start vLLM with `serve_vllm.sh`: from v0.4 the model needs the settings it sets. Started otherwise, vLLM still answers, but not the way the model was trained. Revisions up
to v0.3 are served with prefix caching as before. In the transformers engine `--batch N` (default 4) is the number of prompts
per forward.

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
  "route":  {"type": "choice", "choice": "refund", "confidence": 0.37,
             "probabilities": {"refund": 0.58, "escalate": 0.37, "ignore": 0.05}},
  "urgent": {"type": "noul", "noul": 0.06},
  "anger":  {"type": "score", "score": 0.09, "confidence": 0.91,
             "probabilities": {"0": 0.92, "1": 0.07, "2": 0.01}}}}
```

## The three question types

| type | you send | you get |
| --- | --- | --- |
| `choice` | `criteria`: option name -> description (or `null`: the name is then the description), or a list of names. 2 to 255 options | the likeliest option, a probability for each, `confidence` |
| `score` | `criteria`: the ordered list of levels, lowest first | the expected level as a number, a probability for each level, `confidence` |
| `noul` | only `instructions`: a yes/no question | `noul`: the probability that the answer is yes, committed by default (see "Yes/no answers") |

- `state` can be a string or any JSON value. Put there what the model should judge, and the goal or rules it should judge by.
- `instructions` is the question. Several questions in one request are answered independently.
- `confidence` follows the TypeSafe definitions (choice: `(p_max - 1/n) / (1 - 1/n)`): 0 at chance, 1 when certain.
  `probabilities` are always returned, so you can apply your own threshold.

## Yes/no answers (`--noul-floor`)

A `noul` answer is a probability that the answer is yes. By default the server commits it: a probability closer to 0.5 than
the floor moves to the floor on its own side, so 0.62 is returned as 0.801 and 0.30 as 0.199. Answers already outside
stay as they are, and the decision (yes or no) never changes. The model weights are not involved; it is a step in `server.py`.

- **Why.** A caller, or a benchmark, that reads anything between 0.2 and 0.8 as "no answer" would otherwise throw away
  decisions the model did make. JevBench scores yes/no answers this way.
- **Turn it off** with `--noul-floor 0` when you want the model's own probability, for example to rank items by it or to set
  your own threshold. `choice` and `score` answers are never changed.

## Option order, and reading twice (`--gate`)

**The problem.** The model has a slight preference for some option positions, so the order in which you list the options can change the answer. On most questions it does not. It does when the
model is torn between two options, and when there are many options that look alike. Asking 300 questions once as written and
once with the options reversed, the answer changed on 6% of them.

**What does not pay.** Reading every question in several orders and averaging: four orders cost four times as much, helped
only on questions with many similar options, and changed nothing on ordinary decisions, where the model was already sure.

**What we do.** The server reads a question once. Only if its top probability is below the threshold (`--gate`, 0.5 by
default) is it read a second time with the options reversed, and the two readings are averaged. `--gate 0` reads every
question once.

| test | single read | `--gate 0.5` |
| --- | --- | --- |
| 300 questions with 10-26 similar options (legal placements in a Tetris position) | 67.0% | 76.0% |
| many-option questions, 101-255 options | 89.4% | 92.9% |

Measured during development on an earlier checkpoint (same architecture). On ordinary decisions the second
reading changes little: the released model scores 88.3% on the public JevBench items without it and 89.6% with it.

- **It matters** for questions with many similar options: dozens of candidates, moves in a game, long category lists.
- **Turn it off** (`--gate 0`) when tokens matter more than those cases: on ordinary decisions it changes little.
- **Cost.** About a third more tokens on typical traffic. A question that is read twice takes twice as long; on the Tetris
  questions above about half were.
- The threshold is compared with the top probability, not with the `confidence` field.
- With `--gate 0` you can do the same check yourself: ask the question twice with the options in a different order. If the
  answer changes, treat it as "not sure".

## Limits

- Confidence can be off on unfamiliar domains.
- Arithmetic, date arithmetic and multi-step computation are unreliable: compute in code and put the result in the state.
- It answers the question it is asked from the state it is given. It has no memory between requests and takes no actions.
- Several questions about the same state are not cheaper than separate requests.

## Files

| file | what it is |
| --- | --- |
| `prompt.py` | the prompt text and label tokens the model was trained with |
| `serve_vllm.sh` | starts vLLM with the settings the model needs, then the front end |
| `wire.py` | option handling and the answer shape (probabilities, confidence) |
| `server.py` | the HTTP front end |
| `local.py` | the transformers engine (`--engine transformers`) |
| `test_*.py` | `python -m unittest test_wire.py test_prompt.py test_server.py` (no GPU, no model needed) |

## Licence

- Code in this repository: Apache-2.0 (see `LICENSE`).
- Model weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), free for research and evaluation. For a
  commercial licence write to agentmessier.ai@gmail.com.
- The weights are a fine-tuned version of `Qwen/Qwen3.5-4B` (Apache-2.0).
