"""Synthetic random/identity Qwen3.5 fixtures. NOT GLiDE weights or quality evidence."""
import json
import os
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(os.environ.get("LLAMA_CPP_DIR", Path(__file__).resolve().parents[2] / "llama.cpp"))
sys.path.insert(0, str(SOURCE / "gguf-py"))
import gguf

READOUTS = list(range(255))
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CONFIG_SHA256 = "df40c569d6248a8562177ea8377d47c6fbe558521f60dd14e4e8a78d04e7ce10"


def create_fixture(path, identity=False, override=None, omit=None, *, deep_grouped=False, production_shape=False):
    """Identity fixture has zero residual branches, permitting analytic last-row logits."""
    override = override or {}
    omit = set(omit or [])
    writer = gguf.GGUFWriter(path, "qwen35")
    metadata = {
        "glide.format_version": 1,
        "glide.attention_mode": "noncausal_full_attention",
        "glide.prompt_format": "glide-v1",
        "glide.pooling": "last",
        "glide.source_revision": "1a9a7ed94379c2ea6f324176c9bc22fb9b7b7d60",
        "glide.text_only": True,
        "glide.readout_dtype": "BF16",
        "glide.readout_layout": "cls.output",
        "glide.readout_token_ids_json": json.dumps(READOUTS, separators=(",", ":")),
        "glide.tokenizer_sha256": TOKENIZER_SHA256,
        "glide.decision_config_sha256": CONFIG_SHA256,
        "glide.llama_cpp_commit": "8e5a58df87ab59dfceb5b961ae26691e39f30c24",
        "glide.weight_source": "original-bf16",
        "glide.provenance_json": json.dumps({"kind": "original-bf16", "revision": "1a9a7ed94379c2ea6f324176c9bc22fb9b7b7d60"}, separators=(",", ":")),
        "qwen35.attention.causal": False,
        "qwen35.embedding_length_out": 255,
        "qwen35.pooling_type": int(gguf.PoolingType.LAST),
    }
    metadata.update(override)
    for key, value in metadata.items():
        if key in omit:
            continue
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, value)
    writer.add_name("SYNTHETIC TEST FIXTURE, NOT GLiDE")
    writer.add_bool("test.synthetic_fixture", True)
    writer.add_context_length(1024 if production_shape else 256)
    writer.add_embedding_length(32)
    # Deeper fixture makes earlier full attention affect later recurrent/full layers.
    recurrent = [True, False, True, False] if (deep_grouped or production_shape) else [True, False]
    key_heads, value_heads = (2, 6) if (deep_grouped or production_shape) else (2, 2)
    q_heads, kv_heads, head_dim = (24, 4, 256) if production_shape else (2, 2, 16)
    state_size = 16
    inner_size = state_size * value_heads
    conv_size = state_size * (key_heads * 2 + value_heads)
    writer.add_block_count(len(recurrent))
    writer.add_feed_forward_length(64)
    writer.add_head_count(q_heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_dimension_sections([32, 32, 64, 0] if production_shape else [2, 2, 4, 0])
    writer.add_rope_freq_base(10000)
    if "qwen35.attention.recurrent_layers" not in omit and "qwen35.attention.recurrent_layers" not in override:
        writer.add_recurrent_layers(recurrent)
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_inner_size(inner_size)
    writer.add_ssm_state_size(state_size)
    writer.add_ssm_time_step_rank(value_heads)
    writer.add_ssm_group_count(key_heads)
    writer.add_tokenizer_model("test")
    tokens = [f"tok_{i}" for i in range(512)]
    tokens[511] = "<|image_pad|>"
    writer.add_token_list(tokens)
    writer.add_token_types([int(gguf.TokenType.NORMAL)] * 511 + [int(gguf.TokenType.CONTROL)])
    writer.add_vocab_size(512)
    rng = np.random.default_rng(20261024)

    def random(shape):
        return rng.normal(0, 0.04, shape).astype(np.float32)

    def add(name, shape, value=None, zero=False):
        data = np.full(shape, value, np.float32) if value is not None else random(shape)
        if zero:
            data.fill(0)
        writer.add_tensor(name, data)
        return data

    embeddings = add("token_embd.weight", (512, 32))
    add("output_norm.weight", (32,), value=1)
    head = random((255, 32)) * 8
    head_bytes = gguf.quants.quantize(head, gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("cls.output.weight", head_bytes, raw_dtype=gguf.GGMLQuantizationType.BF16)
    head = gguf.quants.dequantize(head_bytes, gguf.GGMLQuantizationType.BF16)
    for layer in range(len(recurrent)):
        prefix = f"blk.{layer}."
        add(prefix + "attn_norm.weight", (32,), value=1)
        add(prefix + "post_attention_norm.weight", (32,), value=1)
        add(prefix + "ffn_gate.weight", (64, 32))
        add(prefix + "ffn_up.weight", (64, 32))
        add(prefix + "ffn_down.weight", (32, 64), zero=identity)
        if recurrent[layer]:
            add(prefix + "attn_qkv.weight", (conv_size, 32))
            add(prefix + "attn_gate.weight", (inner_size, 32))
            add(prefix + "ssm_conv1d.weight", (conv_size, 4))
            add(prefix + "ssm_dt.bias", (value_heads,), value=-1)
            add(prefix + "ssm_a", (value_heads,), value=-1)
            add(prefix + "ssm_beta.weight", (value_heads, 32))
            add(prefix + "ssm_alpha.weight", (value_heads, 32))
            add(prefix + "ssm_norm.weight", (state_size,), value=1)
            add(prefix + "ssm_out.weight", (32, inner_size), zero=identity)
        else:
            add(prefix + "attn_q.weight", (q_heads * head_dim * 2, 32))
            add(prefix + "attn_k.weight", (kv_heads * head_dim, 32))
            add(prefix + "attn_v.weight", (kv_heads * head_dim, 32))
            add(prefix + "attn_output.weight", (32, q_heads * head_dim), zero=identity)
            add(prefix + "attn_q_norm.weight", (head_dim,), value=1)
            add(prefix + "attn_k_norm.weight", (head_dim,), value=1)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return embeddings, head


if __name__ == "__main__":
    create_fixture(Path(sys.argv[1]), identity="--identity" in sys.argv)
