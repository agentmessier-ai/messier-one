"""Prompt and readout tokens of the decision model. The model was trained on exactly this text; changing a character here
changes its answers.

Options 1-26 are labelled A-Z; options 27-255 as the model directory's sys1_merge.json says. The answer is read from the
logits of these label tokens at one position; nothing is generated.
"""
import json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EXTRA = [f"<o{i}>" for i in range(27, 256)]
PRE = "<|im_start|>user\nRead the state and answer the question by choosing exactly one option.\n\nSTATE:\n"
QPRE = "<|im_start|>user\nRead the question, then the state, and answer by choosing exactly one option.\n\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
AA = []   # the 229 two-letter labels of an AA checkpoint, filled by the Prompter


def label(j):
    return LETTERS[j] if j < 26 else AA[j - 26] if AA else f"<o{j + 1}>"


def logprob_key(j):
    """How vLLM names the label token of the j-th shown option in `top_logprobs` (letters are decoded with their space)."""
    return " " + LETTERS[j] if j < 26 else " " + AA[j - 26] if AA else f"<o{j + 1}>"


def aa_labels(enc):
    one = lambda x: len(enc(x)) == 1
    out = [a + b for a in LETTERS for b in LETTERS if one(a + b) and one(" " + a + b)][:229]
    assert len(out) == 229, f"only {len(out)} two-letter labels are single tokens for this tokenizer"
    return out


class Prompter:
    def __init__(self, tok, max_state=8000, max_total=8600, labels="", layout=""):
        assert layout in ("", "qboth"), layout
        self.enc = lambda x: tok.encode(x, add_special_tokens=False)
        self.max_state, self.max_total, self.layout = max_state, max_total, layout
        self.label_ids = [self.enc(" " + c)[0] for c in LETTERS]
        extra = tok.convert_tokens_to_ids(EXTRA)
        if labels == "AA":
            AA[:] = aa_labels(self.enc)
            self.label_ids += [self.enc(" " + s)[0] for s in AA]
        elif all(isinstance(t, int) and t != tok.unk_token_id for t in extra):   # a 26-option checkpoint has no <o27>.. tokens
            self.label_ids += extra

    def __call__(self, state, qtype, text, keys, candidates, order=None):
        """Token ids of the prompt. state: text; qtype: choice | score | boolean; label j shows option order[j]."""
        enc = self.enc; order = order or list(range(len(keys)))
        st = enc(state)
        if len(st) > self.max_state:   # a state that is too long keeps its beginning and its end
            st = st[: self.max_state // 2] + enc("\n...[truncated]...\n") + st[-self.max_state // 2:]
        opts = "\n".join(f"{label(j)}. {keys[i]}: {candidates[i]}" for j, i in enumerate(order))
        qo = f"QUESTION ({qtype}): {text}\n\nOPTIONS:\n{opts}"
        if self.layout:
            pre = enc(QPRE + qo + "\n\nSTATE:\n")
            post = enc("\n\n" + qo + "\n\nReply with the letter only." + POST)
        else:
            pre = enc(PRE)
            post = enc(f"\n\n{qo}\n\nReply with the letter only." + POST)
        if len(post) > self.max_total - 64:
            post = post[-(self.max_total - 64):]
        if self.layout:   # the question is also on the left, so the room is taken from the state, not from the left end
            room = max(64, self.max_total - len(pre) - len(post))
            if len(st) > room:
                st = st[: room // 2] + enc("\n...[truncated]...\n") + st[-(room // 2 - 8):]
        return (pre + st + post)[-self.max_total:]


def semantic(value, name):
    """A state, option description or score level as the text the model reads: text as is, objects and arrays as compact
    JSON with sorted keys (the serialisation used in training)."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raise ValueError(f"{name} must be nonempty text, an object or an array")
