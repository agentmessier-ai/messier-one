"""In-process engine (pure transformers, no vLLM): one forward per prompt batch, the log-probabilities of the option labels
at the last position.
"""
import threading

import torch


def nc_mask(lens, L, device, dtype=torch.bfloat16):
    """Attention masks for a right-padded batch, as the model directory asks for."""
    keep = torch.arange(L, device=device)[None, :] < torch.as_tensor(lens, device=device)[:, None]
    m = torch.zeros(len(lens), 1, L, L, device=device, dtype=dtype).masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)
    return {"full_attention": m, "linear_attention": None}


class Local:
    def __init__(self, model_dir, label_ids, noncausal, batch=4, device="cuda"):
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": device}).eval()
        self.backbone, self.head = lm.model, lm.lm_head.weight[label_ids].detach().float().clone()
        self.noncausal, self.batch, self.device, self.lock = noncausal, batch, device, threading.Lock()

    @torch.inference_mode()
    def logprobs(self, prompts, k):
        """prompts: token id lists -> per prompt, the log-probability of each of the first k labels, normalised over those k."""
        out = []
        with self.lock:
            for i in range(0, len(prompts), self.batch):
                chunk = prompts[i:i + self.batch]; L = max(map(len, chunk)); lens = [len(p) for p in chunk]
                ids = torch.tensor([p + [p[-1]] * (L - len(p)) for p in chunk], device=self.device)
                kw = {"attention_mask": nc_mask(lens, L, ids.device)} if self.noncausal else {}
                h = self.backbone(input_ids=ids, use_cache=False, **kw).last_hidden_state
                h = h[torch.arange(len(chunk), device=ids.device), torch.tensor(lens, device=ids.device) - 1]
                out += torch.log_softmax(h.float() @ self.head[:k].t(), -1).tolist()
        return out
