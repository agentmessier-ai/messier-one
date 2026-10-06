"""TypeSafe-compatible front end (POST /v1/systemone) for the decision model served by vLLM.

    vllm serve MODEL_DIR --served-model-name m --port 8000 --max-model-len 10240 --max-num-seqs 32 --max-logprobs 256 \\
        --logprobs-mode processed_logprobs --enable-prefix-caching
    python server.py --model-dir MODEL_DIR --vllm http://127.0.0.1:8000 --served m --port 8080

One request = one state and any number of typed questions (choice / score / noul). All questions go to vLLM in one batched
call; vLLM computes one token restricted to the option labels and returns their log-probabilities. Probabilities are
softmax(logprob / T) over the shown options, T per question type from temps.json in the model directory.
--gate G (optional): a question whose top probability is below G is read once more with the options in reverse order and
the two readings are averaged.
"""
import argparse, json, math, os, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

import wire
from prompt import Prompter, logprob_key, semantic

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True); ap.add_argument("--vllm", default="http://127.0.0.1:8000"); ap.add_argument("--served", default="m")
ap.add_argument("--port", type=int, default=8080); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--gate", type=float, default=0.0)
ap.add_argument("--max-state", type=int, default=8000); ap.add_argument("--max-total", type=int, default=8600)
A = ap.parse_args()
P = Prompter(AutoTokenizer.from_pretrained(A.model_dir), A.max_state, A.max_total)
TEMPS = {}
if os.path.exists(os.path.join(A.model_dir, "temps.json")):
    with open(os.path.join(A.model_dir, "temps.json")) as f:
        TEMPS = json.load(f)


def convert(q):
    """A question of the wire contract -> (type as the model was trained on it, option names, option texts)."""
    c = q.get("criteria") or {}
    if q["type"] == "noul":
        return "boolean", ["true", "false"], ["TRUE", "FALSE"]
    if q["type"] == "choice":
        c = wire.options(c)
        return "choice", list(c), [semantic(v, "option") for v in c.values()]
    levels = c if isinstance(c, list) else list(c.values())
    return "score", [str(i) for i in range(len(levels))], [semantic(v, "level") for v in levels]


def read(items):
    """One /v1/completions call for (type, names, prompt ids, order) items -> their answers, in order."""
    k = max(len(it[1]) for it in items)
    if k > len(P.label_ids):
        raise ValueError(f"{k} options, this checkpoint reads at most {len(P.label_ids)}")
    body = {"model": A.served, "prompt": [it[2] for it in items], "max_tokens": 1, "temperature": 1.0, "top_k": -1, "top_p": 1.0, "min_p": 0.0,
            "seed": 0, "logprobs": k, "allowed_token_ids": P.label_ids[:k]}   # sampling filters off: they would zero all but the top label
    req = urllib.request.Request(A.vllm + "/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    choices = sorted(json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"], key=lambda c: c["index"])
    return [finish(typ, names, c["logprobs"]["top_logprobs"][0], order) for (typ, names, _, order), c in zip(items, choices)]


def finish(typ, names, logprobs, order):
    """logprobs: per shown label; label j shows option order[j]. -> the answer, with probabilities keyed by option name."""
    T = TEMPS.get(typ, 1.0); shown = {i: j for j, i in enumerate(order)}
    z = [logprobs.get(logprob_key(shown[i]), -1e9) / T for i in range(len(names))]
    m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return wire.answer(typ, {n: v / s for n, v in zip(names, e)})


def average(answers):
    a = answers[0]
    if len(answers) == 1:
        return a
    if a["type"] == "noul":
        return {"type": "noul", "noul": sum(x["noul"] for x in answers) / len(answers)}
    return wire.answer(a["type"], {k: sum(x["probabilities"][k] for x in answers) / len(answers) for k in a["probabilities"]})


def decide(state, questions):
    items, ids = [], []
    for qid, q in questions.items():
        typ, names, texts = convert(q); order = list(range(len(names)))
        items.append((typ, names, P(state, typ, q.get("instructions", ""), names, texts, order), order, texts, q.get("instructions", ""))); ids.append(qid)
    out = {qid: [a] for qid, a in zip(ids, read([it[:4] for it in items]))}
    if A.gate:
        again = [(qid, it) for qid, it in zip(ids, items) if wire.top_probability(out[qid][0]) < A.gate]
        if again:
            rev = [(it[0], it[1], P(state, it[0], it[5], it[1], it[4], it[3][::-1]), it[3][::-1]) for _, it in again]
            for (qid, _), a in zip(again, read(rev)):
                out[qid].append(a)
    return {qid: average(v) for qid, v in out.items()}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200 if self.path == "/health" else 404, {"status": "ok"} if self.path == "/health" else {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            if not isinstance(body.get("questions"), dict) or not body["questions"]:
                return self._send(400, {"error": "questions must be a nonempty object"})
            state = semantic(body.get("state"), "state"); t = time.time()
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": str(e)[:300]})
        try:
            answers = decide(state, body["questions"])
        except (ValueError, KeyError, TypeError) as e:   # a malformed question
            return self._send(400, {"error": str(e)[:300]})
        except Exception as e:
            return self._send(500, {"error": repr(e)[:300]})
        self._send(200, {"model": body.get("model", A.served), "answers": answers, "latency_ms": (time.time() - t) * 1000})

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 256   # the default of 5 resets connections under concurrent clients


if __name__ == "__main__":
    print(f"front end on {A.host}:{A.port} -> {A.vllm} ({A.served}); temperatures {TEMPS}; up to {len(P.label_ids)} options", flush=True)
    Server((A.host, A.port), H).serve_forever()
