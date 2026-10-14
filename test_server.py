"""python -m unittest test_server.py   (no GPU, no model: the tokenizer and vLLM are replaced by fakes)"""
import json, sys, tempfile, unittest
from unittest import mock

import torch

import local, prompt
from test_prompt import CharTok


def load_server(gate=0.0, temps=None, merge=None, args=(), config=None):
    d = tempfile.mkdtemp()
    json.dump(config or {"text_config": {}}, open(d + "/config.json", "w"))
    if temps:
        json.dump(temps, open(d + "/temps.json", "w"))
    if merge:
        json.dump(merge, open(d + "/sys1_merge.json", "w"))
    sys.modules.pop("server", None)
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=CharTok()), \
         mock.patch.object(sys, "argv", ["server.py", "--model-dir", d, "--gate", str(gate), *args]):
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


class Commit(unittest.TestCase):
    def test_a_yes_no_answer_inside_the_floor_moves_to_it_on_its_own_side_and_nothing_else_changes(self):
        s = load_server()
        self.assertEqual(s.A.noul_floor, 0.801)
        a = {"a": {"type": "noul", "noul": 0.62}, "b": {"type": "noul", "noul": 0.30}, "c": {"type": "noul", "noul": 0.95}, "d": {"type": "noul", "noul": 0.5},
             "e": {"type": "choice", "choice": "x", "probabilities": {"x": 0.6, "y": 0.4}}}
        out = s.commit(a)
        self.assertEqual([out[k]["noul"] for k in "abcd"], [0.801, 1 - 0.801, 0.95, 0.801])
        self.assertEqual(out["e"], a["e"])
        self.assertEqual(s.commit(a, floor=0.0), a)


class FakeBackbone:
    """Hidden state of position t in row b = [b, t, 0]; records every call."""
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, use_cache, **kw):
        self.calls.append((input_ids, use_cache, kw))
        B, L = input_ids.shape
        h = torch.zeros(B, L, 2); h[:, :, 0] = torch.arange(B)[:, None]; h[:, :, 1] = torch.arange(L)[None, :]
        return mock.Mock(last_hidden_state=h)


def fake_local(noncausal, batch=4):
    e = object.__new__(local.Local)
    e.backbone, e.head = FakeBackbone(), torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    e.noncausal, e.batch, e.device, e.lock = noncausal, batch, "cpu", __import__("threading").Lock()
    return e


class NoncausalMask(unittest.TestCase):
    def test_every_position_sees_every_real_token_of_its_row_and_no_padding(self):
        m = local.nc_mask([3, 5], 5, "cpu", torch.float32)
        self.assertEqual(m["linear_attention"], None); f = m["full_attention"]
        self.assertEqual(tuple(f.shape), (2, 1, 5, 5)); low = torch.finfo(torch.float32).min
        self.assertTrue(torch.equal(f[0, 0], torch.tensor([[0, 0, 0, low, low]] * 5, dtype=torch.float32)))   # a query row above the diagonal is open too
        self.assertTrue((f[1] == 0).all())

    def test_the_default_dtype_is_bfloat16_and_an_unpadded_row_is_still_masked_explicitly(self):
        f = local.nc_mask([4], 4, "cpu")["full_attention"]
        self.assertEqual(f.dtype, torch.bfloat16); self.assertTrue((f == 0).all())


class Engine(unittest.TestCase):
    PROMPTS = [[7, 8, 9], [4, 5], [1, 2, 3, 4, 5]]

    def test_noncausal_every_prompt_is_run_whole_with_the_mask_and_without_a_cache(self):
        e = fake_local(True); out = e.logprobs(self.PROMPTS, 2)
        self.assertEqual(len(e.backbone.calls), 1); ids, use_cache, kw = e.backbone.calls[0]
        self.assertFalse(use_cache); self.assertEqual(ids.tolist(), [[7, 8, 9, 9, 9], [4, 5, 5, 5, 5], [1, 2, 3, 4, 5]])
        f = kw["attention_mask"]["full_attention"]; self.assertEqual(tuple(f.shape), (3, 1, 5, 5))
        self.assertEqual([(f[i, 0, 0] == 0).sum().item() for i in range(3)], [3, 2, 5])
        # the label scores are read at each row's own last real token: position 2, 1, 4
        for i, t in enumerate([2, 1, 4]):
            self.assertAlmostEqual(out[i][1] - out[i][0], float(t) - float(i), places=5)
        self.assertAlmostEqual(sum(torch.tensor(out[0]).exp().tolist()), 1.0, places=5)

    def test_causal_gets_no_mask_and_a_big_request_is_split_into_batches(self):
        e = fake_local(False, batch=2); e.logprobs(self.PROMPTS, 3)
        self.assertEqual(len(e.backbone.calls), 2)
        self.assertTrue(all(kw == {} and not uc for _, uc, kw in e.backbone.calls))

    def test_only_the_asked_labels_are_scored(self):
        self.assertEqual(len(fake_local(False).logprobs(self.PROMPTS, 2)[0]), 2)


class Flag(unittest.TestCase):
    def serve(self, merge=None, args=()):
        with mock.patch.object(local, "Local") as L:
            s = load_server(merge=merge, args=("--engine", "transformers", *args))
        return s, L

    def test_defaults_are_vllm_causal_and_no_local_engine(self):
        s = load_server()
        self.assertEqual((s.A.engine, s.A.noncausal, s.NONCAUSAL, s.LOCAL), ("vllm", "auto", False, None))

    def test_the_flag_comes_from_the_model_directory(self):
        s, L = self.serve({"letter_ids": [], "noncausal": True})
        self.assertTrue(s.NONCAUSAL); self.assertEqual(L.call_args.args[2], True)

    def test_a_model_directory_without_the_flag_is_causal(self):
        self.assertFalse(self.serve({"letter_ids": []})[0].NONCAUSAL); self.assertFalse(self.serve()[0].NONCAUSAL)

    def test_the_command_line_overrides_the_model_directory(self):
        self.assertFalse(self.serve({"noncausal": True}, ("--noncausal", "off"))[0].NONCAUSAL)
        self.assertTrue(self.serve({}, ("--noncausal", "on"))[0].NONCAUSAL)

    def test_a_flagged_model_without_its_config_is_refused_on_vllm(self):
        with self.assertRaises(SystemExit):
            load_server(merge={"noncausal": True})

    def test_a_flagged_model_with_its_config_runs_on_vllm(self):
        s = load_server(merge={"noncausal": True}, config={"text_config": {"is_causal": False}})
        self.assertEqual((s.A.engine, s.NONCAUSAL, s.LOCAL), ("vllm", True, None))

    def test_the_layout_comes_from_the_model_directory(self):
        self.assertEqual(load_server(merge={"layout": "qboth"}).P.layout, "qboth"); self.assertEqual(load_server().P.layout, "")

    def test_the_local_engine_answers_through_the_same_wire_shapes(self):
        s, L = self.serve({"noncausal": True})
        L.return_value.logprobs.side_effect = lambda prompts, k: [[-0.1, -3.0] + [-9.0] * (k - 2)] * len(prompts)
        out = s.commit(s.decide("my card was charged twice", Decide.Q))
        self.assertEqual(out["route"]["choice"], "billing"); self.assertGreater(out["urgent"]["noul"], 0.9)
        self.assertEqual(L.return_value.logprobs.call_args.args[1], 2)


class AALabels(unittest.TestCase):
    def tearDown(self):
        prompt.AA.clear()

    def test_two_letter_labels_for_options_beyond_26(self):
        class AATok(CharTok):
            def encode(self, text, add_special_tokens=False):
                t = text.strip()
                return [3000 + sum(ord(c) for c in t)] if len(t) == 2 and t.isalpha() and t.isupper() else super().encode(text)
        p = prompt.Prompter(AATok(), labels="AA")
        self.assertEqual(len(p.label_ids), 255)
        self.assertEqual((prompt.label(26), prompt.logprob_key(26), prompt.logprob_key(25)), ("AA", " AA", " Z"))

    def test_the_default_labels_are_unchanged(self):
        self.assertEqual((prompt.label(26), prompt.logprob_key(26)), ("<o27>", "<o27>"))


if __name__ == "__main__":
    unittest.main()
