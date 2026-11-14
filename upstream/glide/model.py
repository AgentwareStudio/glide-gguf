"""Glide model: loading, batching and the decision readout.

One forward pass per question prompt. Full-attention layers attend bidirectionally over every non-pad token of the prompt;
linear-attention layers keep their recurrence. The hidden state of the last token goes through a 255-way readout; a softmax over the
question's options (scaled by the temperature in decision_config.json) gives the probabilities. Images are encoded by the vision
encoder and placed into the same forward pass.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import time

import numpy as np
import torch

from .images import image_inputs, load_image
from .prompt import MAX_OPTIONS, Prompter, RequestError, options_for, split_state

log = logging.getLogger("glide")

LM_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
DECISION_CONFIG = "decision_config.json"
READOUT_FILE = "readout.safetensors"
DEFAULT_NAME = "glide-v1.2"
DEFAULT_MAX_LEN = 40000          # tokens per question prompt, image tokens included
DEFAULT_TOKEN_BUDGET = 16384     # padded tokens per forward micro-batch


# ----------------------------------------------------------------------------------------------------------------- tensors
def attention_masks(padding):
    """Per-layer-type attention masks for a left-padded batch: full-attention layers see every non-pad token (no causal mask);
    linear-attention layers only zero pad positions."""
    lin = None if bool(padding.all()) else padding
    return {"full_attention": padding[:, None, None, :].bool(), "linear_attention": lin}


def position_ids(padding):
    return (padding.long().cumsum(-1) - 1).clamp_(min=0)


def left_pad(seqs, pad_id, device):
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        m[i, T - len(s):] = 1
    return x.to(device, non_blocking=True), m.to(device, non_blocking=True)


def mrope_positions(input_ids, padding, grid_thw, image_token_id, merge):
    """[4, B, T] position ids (text position, t, h, w) for rows with images: text tokens advance all three axes together; an image's
    merged tokens get (start, start + row, start + col) and the following text starts at start + max(rows, cols). Pads get 0."""
    B, T = input_ids.shape
    pos3 = torch.zeros(3, B, T, dtype=torch.long)
    grids = [tuple(int(v) for v in g) for g in grid_thw.tolist()]
    gi = 0
    ids_all = input_ids.tolist()
    pad_all = padding.bool().tolist()
    for b in range(B):
        ids = [t for t, ok in zip(ids_all[b], pad_all[b]) if ok]
        parts, cur, i, n = [], 0, 0, len(ids)
        while i < n:
            if ids[i] == image_token_id:
                t, gh, gw = grids[gi]
                gi += 1
                lh, lw = gh // merge, gw // merge
                tt = torch.arange(t).view(-1, 1, 1).expand(t, lh, lw).flatten() + cur
                hh = torch.arange(lh).view(1, -1, 1).expand(t, lh, lw).flatten() + cur
                ww = torch.arange(lw).view(1, 1, -1).expand(t, lh, lw).flatten() + cur
                parts.append(torch.stack([tt, hh, ww]))
                cur += max(lh, lw)
                i += t * lh * lw
            else:
                j = i
                while j < n and ids[j] != image_token_id:
                    j += 1
                parts.append(torch.arange(cur, cur + j - i).view(1, -1).expand(3, -1))
                cur += j - i
                i = j
        if parts:
            pos3[:, b, torch.tensor(pad_all[b])] = torch.cat(parts, dim=1)
    if gi != len(grids):
        raise ValueError(f"{len(grids)} images but {gi} image-token runs")
    return torch.cat([position_ids(padding).cpu()[None], pos3], dim=0).to(input_ids.device)


class _Network(torch.nn.Module):
    def __init__(self, text_config, vision_config=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        super().__init__()
        self.backbone = Qwen3_5TextModel(text_config)
        self.readout = torch.nn.Linear(text_config.hidden_size, MAX_OPTIONS, bias=False)
        if vision_config is not None:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
            self.visual = Qwen3_5VisionModel(vision_config)
            self.image_token_id = int(vision_config.image_token_id)
            self.merge = int(vision_config.spatial_merge_size)
        else:
            self.visual = None

    def forward(self, input_ids, padding, pixel_values=None, image_grid_thw=None):
        mask = attention_masks(padding)
        if pixel_values is not None:
            e = self.backbone.embed_tokens(input_ids)
            v = self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
            m = input_ids == self.image_token_id
            if int(m.sum()) != v.shape[0]:
                raise ValueError(f"image tokens {int(m.sum())} != image features {v.shape[0]}")
            e = e.masked_scatter(m[..., None], v.to(e.dtype))
            pos = mrope_positions(input_ids, padding, image_grid_thw, self.image_token_id, self.merge)
            h = self.backbone(inputs_embeds=e, attention_mask=mask, position_ids=pos, use_cache=False).last_hidden_state
        else:
            h = self.backbone(input_ids=input_ids, attention_mask=mask, position_ids=position_ids(padding), use_cache=False).last_hidden_state
        return self.readout(h[:, -1])          # left padding: the last position is every row's final "("


def _restore_rotary(module, device):
    """Rotary-embedding buffers are not stored in the checkpoint: recompute them after meta-device construction."""
    for m in module.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_type"):
            if m.rope_type == "axial" and hasattr(m, "compute_axial_rope_parameters"):
                inv, scale = m.compute_axial_rope_parameters(m.config)
            else:
                fn = m.compute_default_rope_parameters if m.rope_type == "default" else None
                if fn is None:
                    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                    fn = ROPE_INIT_FUNCTIONS[m.rope_type]
                inv, scale = fn(m.config)
            m.inv_freq = inv.to(device, torch.float32)
            m.original_inv_freq = inv.clone().to(device, torch.float32)
            m.attention_scaling = scale


class _WeightReader:
    """Lazy reader over the sharded safetensors checkpoint listed in model.safetensors.index.json."""

    def __init__(self, path):
        from safetensors import safe_open
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        self.files, self.where = {}, {}
        for name in sorted(set(weight_map.values())):
            h = safe_open(os.path.join(path, name), framework="pt", device="cpu")
            self.files[name] = h
            for k in h.keys():
                self.where[k] = name

    def has(self, k):
        return k in self.where

    def get(self, k):
        return self.files[self.where[k]].get_tensor(k)


def _softmax(logits, n, temperature):
    v = [x / temperature for x in logits[:n]]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    z = sum(e)
    return [x / z for x in e]


def _answer(qtype, keys, probs):
    pmap = dict(zip(keys, probs))
    top = max(pmap, key=pmap.get)
    conf = pmap[top] - sorted(pmap.values())[-2] if len(pmap) > 1 else pmap[top]
    if qtype == "noul":
        return {"type": "noul", "noul": pmap["true"], "confidence": abs(pmap["true"] - 0.5) * 2}
    if qtype == "score":
        return {"type": "score", "score": int(top), "expected_level": sum(float(k) * p for k, p in pmap.items()), "confidence": conf,
                "probabilities": pmap}
    return {"type": "choice", "choice": top, "confidence": conf, "probabilities": pmap}


def _read_local_image(x):
    """Python API convenience: a path to an image file is read from disk; anything else is passed through."""
    if isinstance(x, os.PathLike) or (isinstance(x, str) and not x.startswith("data:") and len(x) < 4096 and os.path.isfile(x)):
        with open(x, "rb") as f:
            return f.read()
    return x


# ----------------------------------------------------------------------------------------------------------------- model
class Glide:
    """Decision model. Load with Glide.from_pretrained(path_or_repo_id); call decide() or decide_batch()."""

    def __init__(self, network, tokenizer, config, device="cuda", max_len=DEFAULT_MAX_LEN, token_budget=DEFAULT_TOKEN_BUDGET, name=DEFAULT_NAME):
        self.network, self.tokenizer, self.config, self.device = network, tokenizer, config, device
        self.max_len, self.token_budget, self.name = int(max_len), int(token_budget), name
        self.codes = list(config["codes"])
        self.temperature = float(config.get("temperature", 1.0))
        self.prompter = Prompter(tokenizer, self.codes)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if self.network.visual is not None:
            vc = self.network.visual.config
            self._patch = (int(vc.patch_size), int(vc.temporal_patch_size), int(vc.spatial_merge_size))

    @property
    def vision(self):
        return self.network.visual is not None

    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=None, vision=True, max_len=DEFAULT_MAX_LEN, token_budget=DEFAULT_TOKEN_BUDGET,
                        name=DEFAULT_NAME, revision=None, token=None):
        """path: a local model directory or a Hugging Face repo id. vision=False skips the vision encoder (text-only serving;
        requests with images are then rejected)."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
        t0 = time.time()
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision, token=token)
        torch.backends.cuda.enable_cudnn_sdp(False)          # masked SDPA runs on the memory-efficient kernel
        dtype = dtype or torch.bfloat16
        with open(os.path.join(path, DECISION_CONFIG)) as f:
            config = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(path)
        _check_codes(tokenizer, config)
        full = AutoConfig.from_pretrained(path)
        text_config = getattr(full, "text_config", full)
        text_config._attn_implementation = "sdpa"
        text_config.use_cache = False
        reader = _WeightReader(path)
        vision_config = None
        if vision and getattr(full, "vision_config", None) is not None and any(k.startswith(VISION_PREFIX) for k in reader.where):
            vision_config = full.vision_config
            vision_config._attn_implementation = "sdpa"
            vision_config.image_token_id = int(full.image_token_id)
        with torch.device("meta"):
            net = _Network(text_config, vision_config).to(dtype)
        net = net.to_empty(device=device)
        missing = []
        with torch.no_grad():
            for prefix, module in ((LM_PREFIX, net.backbone), (VISION_PREFIX, net.visual)):
                if module is None:
                    continue
                for n, p in module.named_parameters():
                    k = prefix + n
                    if not reader.has(k):
                        missing.append(k)
                        continue
                    p.copy_(reader.get(k).to(device=device, dtype=dtype))
            net.readout.weight.copy_(load_file(os.path.join(path, READOUT_FILE))["weight"].to(device=device, dtype=dtype))
        if missing:
            raise RuntimeError(f"missing weights: {missing[:5]} ({len(missing)} total)")
        _restore_rotary(net, device)
        net.eval().requires_grad_(False)
        log.info("loaded %s in %.0f s (vision=%s)", path, time.time() - t0, vision_config is not None)
        return cls(net, tokenizer, config, device=device, max_len=max_len, token_budget=token_budget, name=name)

    # ------------------------------------------------------------------------------------------------------------- public API
    def decide(self, state, questions, images=None):
        """One request. state: string, JSON value, or a multimodal parts list; questions: {key: question};
        images: optional list of PIL images, file paths, bytes, base64 strings or data URLs (placed before the state text).
        Returns {"model", "answers", "usage"}; raises RequestError for requests the model cannot serve."""
        req = {"state": state, "questions": questions}
        if images:
            req["images"] = [_read_local_image(x) for x in images]
        r = self.decide_batch([req])[0]
        if isinstance(r, RequestError):
            raise r
        return r

    def decide_batch(self, requests):
        """Several requests scored together. requests: list of {"state", "questions", optional "images"}.
        Returns one item per request: a response dict, or a RequestError instance (not raised)."""
        results = [None] * len(requests)
        prepared = []
        for ri, req in enumerate(requests):
            try:
                entries, vis = self._prepare(req)
                prepared.append((ri, entries, vis))
            except RequestError as e:
                results[ri] = e
        texts = [e[3] for _, entries, _ in prepared for e in entries]
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
        seqs, vis_list, slots, usage, k = [], [], [], {}, 0
        for ri, entries, vis in prepared:
            req_ids = ids[k:k + len(entries)]
            k += len(entries)
            if any(len(x) + 8 > self.max_len for x in req_ids):
                results[ri] = RequestError("prompt exceeds the maximum context length")
                continue
            for e, x in zip(entries, req_ids):
                slots.append((ri, e))
                seqs.append(x)
                vis_list.append(vis)
            usage[ri] = sum(len(x) for x in req_ids)
        logits = self._logits(seqs, vis_list) if seqs else []
        answers = collections.defaultdict(dict)
        for (ri, (qk, qtype, keys, _)), row in zip(slots, logits):
            answers[ri][qk] = _answer(qtype, keys, _softmax(row, len(keys), self.temperature))
        for ri in range(len(requests)):
            if results[ri] is None:
                results[ri] = {"model": self.name, "answers": answers[ri], "usage": {"input_tokens": usage.get(ri, 0), "output_tokens": 0}}
        return results

    # ------------------------------------------------------------------------------------------------------------- internals
    def _prepare(self, req):
        """-> ([(question key, type, option keys, prompt text)], image inputs or None)."""
        if not isinstance(req, dict):
            raise RequestError("request must be an object")
        questions = req.get("questions")
        if not isinstance(questions, dict):
            raise RequestError("questions must be an object mapping keys to questions")
        state, refs = split_state(req.get("state"))
        refs = list(refs) + list(req.get("images") or [])
        vis = None
        if refs:
            if not self.vision:
                raise RequestError("images are not supported by this deployment")
            patch, temporal, merge = self._patch
            try:
                vis = [image_inputs(load_image(u), patch=patch, temporal=temporal, merge=merge) for u in refs]
            except Exception as e:  # noqa: BLE001
                raise RequestError(f"invalid image: {type(e).__name__}: {str(e)[:120]}") from None
        ntok = [v[2] for v in vis] if vis else None
        entries = []
        for qk, q in questions.items():
            opts = options_for(q)
            if len(opts) > MAX_OPTIONS:
                raise RequestError(f"options per choice exceeds the label set ({len(opts)} > {MAX_OPTIONS})")
            entries.append((qk, q["type"], [key for key, _ in opts], self.prompter.text(state, q, opts, ntok)))
        return entries, vis

    def _logits(self, seqs, vis):
        """Token-id lists (+ per-sequence image inputs or None) -> [255] logits per sequence. Sequences are length-sorted and packed
        into left-padded micro-batches under the padded-token budget; text-only and image sequences are batched separately."""
        res = [None] * len(seqs)
        with torch.inference_mode():
            for group in ([k for k in range(len(seqs)) if not vis[k]], [k for k in range(len(seqs)) if vis[k]]):
                order = sorted(group, key=lambda i: len(seqs[i]))
                i = 0
                while i < len(order):
                    j = i + 1
                    while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= self.token_budget:
                        j += 1
                    idx = order[i:j]
                    x, m = left_pad([seqs[k] for k in idx], self.pad_id, self.device)
                    if vis[idx[0]]:
                        pv = torch.from_numpy(np.concatenate([p for k in idx for p, _, _ in vis[k]])).to(self.device)
                        g = torch.tensor([gr for k in idx for _, gr, _ in vis[k]], device=self.device)
                        out = self.network(x, m, pixel_values=pv, image_grid_thw=g)
                    else:
                        out = self.network(x, m)
                    for k, v in zip(idx, out.float().cpu().tolist()):
                        res[k] = v
                    i = j
        return res


def _check_codes(tokenizer, config):
    codes, ids = config.get("codes") or [], config.get("token_ids") or []
    if len(codes) != MAX_OPTIONS or len(ids) != MAX_OPTIONS:
        raise ValueError(f"{DECISION_CONFIG} must list {MAX_OPTIONS} codes and token ids")
    for c, i in zip(codes, ids):
        if tokenizer.encode(c, add_special_tokens=False) != [i]:
            raise ValueError(f"label code {c!r} does not match the tokenizer")
