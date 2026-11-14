"""Integration test with synthetic 2-layer hybrid model, never a GLiDE parity claim."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from tiny_fixture import create_fixture, READOUTS, gguf

binary = str(Path(sys.argv[1]).resolve())

def request(identifier, tokens):
    return {"id": identifier, "token_ids": tokens, "readout_token_ids": READOUTS}

def run(model, inputs, *extra):
    proc = subprocess.run([binary, "--model", str(model), "--ctx-size", "64", "--threads", "2", *extra],
        input="\n".join(json.dumps(x) if not isinstance(x, str) else x for x in inputs) + "\n",
        text=True, capture_output=True, timeout=60)
    if proc.returncode:
        raise AssertionError(f"runner exited {proc.returncode}:\n{proc.stderr[-10000:]}")
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(rows) == len(inputs), (len(rows), len(inputs), proc.stdout)
    ready = [json.loads(line) for line in proc.stderr.splitlines() if line.startswith('{') and '"event":"ready"' in line]
    assert len(ready) == 1 and ready[0]["causal_attention"] is False
    assert ready[0]["n_batch"] >= 64 and ready[0]["n_ubatch"] >= 64
    return rows

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    identity = root / "identity.gguf"
    embeddings, head = create_fixture(identity, identity=True)
    rows = run(identity, [request("last", [1, 7, 9]), request("single", [9]), request("other", [1, 7])])
    analytic_max_abs_error = 0.0
    for row, token in zip(rows, [9, 9, 7]):
        assert row["ok"], row
        x = embeddings[token]
        # The pinned CPU BF16 matmul uses BF16 RHS activations (vec_dot_type=BF16).
        hidden = x / np.sqrt(np.mean(x * x) + 1e-6)
        hidden = gguf.quants.dequantize(gguf.quants.quantize(hidden, gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16)
        expected = head @ hidden
        analytic_max_abs_error = max(analytic_max_abs_error, float(np.max(np.abs(np.asarray(row["logits"]) - expected))))
        np.testing.assert_allclose(row["logits"], expected, rtol=2e-5, atol=2e-5)
        assert len(row["logits"]) == 255
        assert any(x < 0 for x in row["logits"]), "Must return raw logits, not softmax probabilities"
        assert abs(np.linalg.norm(row["logits"]) - 1) > 1, "Must not normalize vectors"
    np.testing.assert_array_equal(rows[0]["logits"], rows[1]["logits"])

    random = root / "random.gguf"
    create_fixture(random)
    bad_ids = request("mismatch", [1]); bad_ids["readout_token_ids"] = list(reversed(READOUTS))
    inputs = [request("a1", [1, 2, 3, 4]), "{bad}", request("b", [3, 5, 7]),
              request("a2", [1, 2, 3, 4]), request("empty", []), request("too-long", [1] * 65),
              request("multimodal", [511]), bad_ids, request("a3", [1, 2, 3, 4])]
    rows = run(random, inputs)
    for index in [0, 2, 3, 8]:
        assert rows[index]["ok"], rows[index]
        assert np.isfinite(rows[index]["logits"]).all()
    for index, code in [(1, "invalid_json"), (4, "empty_prompt"), (5, "prompt_too_long"), (6, "multimodal_unsupported"), (7, "readout_mismatch")]:
        assert not rows[index]["ok"] and rows[index]["error"]["code"] == code, rows[index]
    np.testing.assert_array_equal(rows[0]["logits"], rows[3]["logits"])
    np.testing.assert_array_equal(rows[0]["logits"], rows[8]["logits"])
    assert rows[0]["logits"] != rows[2]["logits"]

    # Request-size/line recovery at the actual subprocess boundary, not just a parser unit.
    stream_rows = run(random, [" " * (4 * 1024 * 1024 + 1), request("after-oversize", [1, 2, 3, 4]),
        "", request("at-limit", [1] * 64), request("one", [1])])
    assert stream_rows[0]["error"]["code"] == "request_too_large"
    assert stream_rows[2]["error"]["code"] == "invalid_json"
    assert stream_rows[3]["ok"] and stream_rows[3]["prompt_tokens"] == 64
    assert stream_rows[4]["ok"] and stream_rows[4]["prompt_tokens"] == 1
    np.testing.assert_array_equal(stream_rows[1]["logits"], rows[0]["logits"])

    invalid_metadata = [({}, ["qwen35.attention.recurrent_layers"]),
        ({"qwen35.attention.recurrent_layers": "false"}, []),
        ({"qwen35.attention.recurrent_layers": [1, 0]}, []),
        ({"qwen35.attention.recurrent_layers": [True]}, []),
        ({"qwen35.attention.recurrent_layers": [True, True]}, []),({"qwen35.attention.causal": True}, []), ({"glide.text_only": False}, []),
        ({"glide.format_version": 2}, []), ({"glide.readout_layout": "vocab"}, []),
        ({"glide.tokenizer_sha256": "0" * 64}, []), ({"glide.decision_config_sha256": "0" * 64}, []),
        ({"qwen35.decision.type": "d3"}, []), ({"decision.type": "d3"}, []),
        ({"clip.has_vision_encoder": True}, []), ({"glide.unknown": True}, []),
        ({}, ["glide.pooling"]), ({"qwen35.embedding_length_out": 256}, []),
        ({"glide.llama_cpp_commit": "0" * 40}, []), ({"glide.weight_source": "unverified"}, []),
        ({"glide.provenance_json": "[]"}, []), ({"glide.provenance_json": "{bad}"}, []),
        ({"glide.provenance_json": " " * 16385}, []), ({"glide.provenance_json": "{}"}, [])]
    for index, (override, omit) in enumerate(invalid_metadata):
        invalid = root / f"bad-{index}.gguf"
        create_fixture(invalid, override=override, omit=omit)
        proc = subprocess.run([binary, "--model", str(invalid)], capture_output=True, text=True, timeout=10)
        assert proc.returncode == 2 and not proc.stdout
        assert json.loads(proc.stderr.splitlines()[-1])["code"] == "unsupported_model", proc.stderr
print(json.dumps({"result": "passed", "fixture": "synthetic 2-layer hybrid Qwen3.5, NOT GLiDE", "request_response_cases": 17, "metadata_rejection_cases": len(invalid_metadata), "analytic_max_abs_error": analytic_max_abs_error, "raw_logits": 255, "aba_state_reset": "bitwise_equal", "gpu_tested": False}))
