"""python -m unittest test_server.py   (no GPU, no model: the tokenizer and vLLM are replaced by fakes)"""
import json, sys, tempfile, unittest
from unittest import mock

from test_prompt import CharTok


def load_server(gate=0.0, temps=None):
    d = tempfile.mkdtemp()
    if temps:
        json.dump(temps, open(d + "/temps.json", "w"))
    sys.modules.pop("server", None)
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=CharTok()), \
         mock.patch.object(sys, "argv", ["server.py", "--model-dir", d, "--gate", str(gate)]):
        import server
    return server


def fake_vllm(server, logprob_of):
    """Replaces the HTTP call: logprob_of(prompt text) -> {label key: logprob}."""
    seen = []
    def urlopen(req, timeout=None):
        body = json.loads(req.data); seen.append(body)
        ch = [{"index": i, "logprobs": {"top_logprobs": [logprob_of(CharTok().text(p))]}} for i, p in enumerate(body["prompt"])]
        return mock.Mock(read=lambda: json.dumps({"choices": ch[::-1]}).encode())   # out of order, as a batch may come back
    return mock.patch.object(server.urllib.request, "urlopen", urlopen), seen


class Convert(unittest.TestCase):
    def test_the_three_question_types(self):
        s = load_server()
        self.assertEqual(s.convert({"type": "noul", "instructions": "ok?"}), ("boolean", ["true", "false"], ["TRUE", "FALSE"]))
        self.assertEqual(s.convert({"type": "choice", "criteria": {"a": "first", "b": None}}), ("choice", ["a", "b"], ["first", "b"]))
        self.assertEqual(s.convert({"type": "score", "criteria": ["low", "high"]}), ("score", ["0", "1"], ["low", "high"]))


class Decide(unittest.TestCase):
    Q = {"route": {"type": "choice", "instructions": "Where?", "criteria": {"billing": "money", "tech": "bugs"}}, "urgent": {"type": "noul", "instructions": "Urgent?"}}

    def test_all_questions_go_out_in_one_call_and_come_back_keyed_by_name(self):
        s = load_server(); patch, seen = fake_vllm(s, lambda text: {" A": -0.1, " B": -3.0})
        with patch:
            out = s.decide("my card was charged twice", self.Q)
        self.assertEqual(len(seen), 1); self.assertEqual(len(seen[0]["prompt"]), 2); self.assertEqual(seen[0]["allowed_token_ids"], s.P.label_ids[:2])
        self.assertEqual(out["route"]["choice"], "billing"); self.assertAlmostEqual(sum(out["route"]["probabilities"].values()), 1.0)
        self.assertEqual(out["urgent"]["type"], "noul"); self.assertGreater(out["urgent"]["noul"], 0.9)

    def test_temperature_per_type_from_the_model_directory(self):
        lp = lambda text: {" A": 0.0, " B": -1.0}
        cold, hot = load_server(temps={"choice": 0.5}), load_server(temps={"choice": 2.0})
        for s in (cold, hot):
            patch, _ = fake_vllm(s, lp)
            with patch:
                s.p = s.decide("x", {"q": self.Q["route"]})["q"]["probabilities"]["billing"]
        self.assertGreater(cold.p, hot.p)

    def test_gate_reads_an_unsure_question_again_reversed_and_averages(self):
        s = load_server(gate=0.9)
        def lp(text):   # the model likes whatever is shown first: only the average is order-free
            return {" A": -0.5, " B": -1.0}
        patch, seen = fake_vllm(s, lp)
        with patch:
            out = s.decide("x", {"q": self.Q["route"]})
        self.assertEqual(len(seen), 2); self.assertAlmostEqual(out["q"]["probabilities"]["billing"], 0.5)

    def test_a_sure_question_is_read_once(self):
        s = load_server(gate=0.5); patch, seen = fake_vllm(s, lambda text: {" A": 0.0, " B": -9.0})
        with patch:
            s.decide("x", {"q": self.Q["route"]})
        self.assertEqual(len(seen), 1)

    def test_more_options_than_the_checkpoint_has_labels_is_refused(self):
        s = load_server(); s.P.label_ids = s.P.label_ids[:26]
        with self.assertRaises(ValueError):
            s.decide("x", {"q": {"type": "choice", "criteria": {f"k{i}": None for i in range(27)}}})


if __name__ == "__main__":
    unittest.main()
