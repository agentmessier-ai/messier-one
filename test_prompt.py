"""python -m unittest test_prompt.py   (no GPU, no model)"""
import unittest

from prompt import LETTERS, Prompter, label, logprob_key, semantic


class CharTok:
    """One token per character, except that the special markers and the labels <o27>.. are single tokens."""
    SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] + [f"<o{i}>" for i in range(27, 256)]
    unk_token_id = 0

    def __init__(self, extra=True):
        self.extra = extra

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            sp = next((s for s in self.SPECIAL if text.startswith(s, i)), None)
            if sp:
                out.append(1000 + self.SPECIAL.index(sp)); i += len(sp)
            elif text[i] == " " and i + 1 < len(text) and text[i + 1] in LETTERS and len(text) == 2:
                out.append(500 + LETTERS.index(text[i + 1])); i += 2      # " A" as the tokenizer's single label token
            else:
                out.append(2000 + ord(text[i])); i += 1
        return out

    def convert_tokens_to_ids(self, toks):
        return [1000 + self.SPECIAL.index(t) if self.extra else self.unk_token_id for t in toks]

    def text(self, ids):
        return "".join(self.SPECIAL[i - 1000] if 1000 <= i < 2000 else chr(i - 2000) for i in ids)


class Prompt(unittest.TestCase):
    def test_the_prompt_is_the_text_the_model_was_trained_on(self):
        tok = CharTok(); p = Prompter(tok)
        ids = p("order 8841, headphones dead", "choice", "What now?", ["refund", "escalate"], ["refund the order", "send to a human"])
        self.assertEqual(tok.text(ids), "<|im_start|>user\nRead the state and answer the question by choosing exactly one option.\n\nSTATE:\n"
                         "order 8841, headphones dead\n\nQUESTION (choice): What now?\n\nOPTIONS:\nA. refund: refund the order\nB. escalate: send to a human\n\n"
                         "Reply with the letter only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")

    def test_an_order_changes_which_option_each_label_shows(self):
        tok = CharTok(); p = Prompter(tok)
        self.assertIn("A. b: second\nB. a: first", tok.text(p("s", "choice", "q", ["a", "b"], ["first", "second"], [1, 0])))

    def test_options_beyond_26_use_the_extra_label_tokens(self):
        tok = CharTok(); p = Prompter(tok); names = [f"k{i}" for i in range(30)]
        text = tok.text(p("s", "choice", "q", names, names))
        self.assertIn("Z. k25: k25\n<o27>. k26: k26", text); self.assertEqual(len(p.label_ids), 255)
        self.assertEqual((label(25), label(26), logprob_key(0), logprob_key(26)), ("Z", "<o27>", " A", "<o27>"))

    def test_a_checkpoint_without_the_extra_tokens_reads_26_options(self):
        self.assertEqual(len(Prompter(CharTok(extra=False)).label_ids), 26)

    def test_a_long_state_keeps_its_beginning_and_end(self):
        tok = CharTok(); p = Prompter(tok, max_state=20, max_total=4000)
        text = tok.text(p("a" * 50 + "b" * 50, "boolean", "q", ["true", "false"], ["TRUE", "FALSE"]))
        self.assertIn("a" * 10 + "\n...[truncated]...\n" + "b" * 10, text); self.assertNotIn("a" * 11, text)

    def test_the_whole_prompt_is_cut_from_the_front_so_the_question_survives(self):
        tok = CharTok(); p = Prompter(tok, max_state=5000, max_total=300)
        ids = p("x" * 1000, "boolean", "Is it?", ["true", "false"], ["TRUE", "FALSE"])
        self.assertEqual(len(ids), 300); self.assertTrue(tok.text(ids).endswith("Answer:")); self.assertIn("Is it?", tok.text(ids))


class Semantic(unittest.TestCase):
    def test_text_stays_and_structures_become_compact_sorted_json(self):
        self.assertEqual(semantic("as is", "state"), "as is")
        self.assertEqual(semantic({"b": 1, "a": ["x", "é"]}, "state"), '{"a":["x","é"],"b":1}')

    def test_nothing_to_read_is_an_error(self):
        for bad in (None, "", "   ", 3):
            with self.assertRaises(ValueError):
                semantic(bad, "state")


if __name__ == "__main__":
    unittest.main()
