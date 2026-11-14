"""Production FA head geometry on synthetic weights. CPU pass is NOT CUDA proof.

Produces control_result.json plus complete fixtures, requests, stdout/stderr.
A supervisor may enrich the result with independently observed GPU cleanliness;
this script NEVER fabricates such proof and NEVER calls nvidia-smi itself.
"""
if not __debug__:
    raise RuntimeError("Synthetic admission control refuses optimized Python: assertions must remain active")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from tiny_fixture import create_fixture, READOUTS, gguf

PROFILE = "glide-fa-hostemb-v1"
CONTRACT_SHA = "e9b7c3b70d9cd5af8ce776ffa8e955f1c23d9cef50c84b66a14fd72ae599a433"
SHAPE = {"full_attention_head_dim": 256, "query_heads": 24, "kv_heads": 4,
         "full_attention_layers": 2, "hidden_size": 32, "feed_forward_size": 64,
         "recurrent_key_heads": 2, "recurrent_value_heads": 6, "recurrent_group_ratio": "1:3"}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def request(identifier, tokens):
    return {"id": identifier, "token_ids": tokens, "readout_token_ids": READOUTS}

def validate_evidence(row, tokens, backend):
    assert row.get("ok") is True, row
    ev = row["execution_evidence"]
    assert ev["schema"] == "glide.native_execution_evidence.v1" and ev["backend_profile"] == PROFILE
    assert ev["requested_backend"] == backend and ev["decode_calls"] == 1 and ev["prompt_tokens"] == len(tokens)
    assert ev["expected_full_attention_layers"] == 2
    assert ev["state_reset_before"] is True and ev["state_reset_after"] is True
    assert not ev["vocabulary_matmul_nodes"]
    assert len(ev["flash_attention_nodes"]) == 2
    assert len(set(x["name"] for x in ev["flash_attention_nodes"])) == 2
    expected_backend = backend.upper()
    def device(record, cpu=False):
        assert set(record) == {"backend", "type", "name", "buffer"} and record["name"] and record["buffer"]
        assert record["backend"] == ("CPU" if cpu else expected_backend)
        assert record["type"] in (["CPU"] if cpu or backend == "cpu" else ["GPU", "IGPU"])
    for group, op, weight in [("flash_attention_nodes", "FLASH_ATTN_EXT", None),
                             ("classifier_nodes", "MUL_MAT", "cls.output.weight"),
                             ("token_embedding_nodes", "GET_ROWS", "token_embd.weight")]:
        nodes = ev[group]
        if weight: assert len(nodes) == 1
        for node in nodes:
            assert node["name"] and node["op"] == op and node["phase"] == "post_execution" and node["executed"] is True
            if weight == "token_embd.weight":
                out = node["output_device"]
                assert (out["backend"], out["type"]) in [("CPU", "CPU"), ("CUDA", "GPU"), ("CUDA", "IGPU")]
            else: device(node["output_device"])
            if weight:
                assert node["weight_name"] == weight
                device(node["weight_device"], cpu=weight == "token_embd.weight")
    assert len(row["logits"]) == 255 and np.isfinite(row["logits"]).all()
    return ev

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--control", required=True, help="CPU non-FA causal/noncausal control executable")
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave an earlier success receipt at this path.
    (out / "control_result.json").write_text(json.dumps({"schema": "glide.native_fa_synthetic_control_result.v1", "status": "running"}) + "\n")
    runner, control = str(Path(args.runner).resolve()), str(Path(args.control).resolve())
    model, identity = out / "production-shape-SYNTHETIC.gguf", out / "identity-SYNTHETIC.gguf"
    create_fixture(model, production_shape=True)
    embeddings, head = create_fixture(identity, production_shape=True, identity=True)
    runs = []

    def invoke(tag, binary, argv, requests, native=False, capacity=256):
        source = "".join((x if isinstance(x, str) else json.dumps(x, separators=(",", ":"))) + "\n" for x in requests)
        input_path, stdout_path, stderr_path = [out / f"{tag}.{kind}" for kind in ["input.jsonl", "stdout.jsonl", "stderr.log"]]
        input_path.write_text(source)
        # subprocess.run waits/reaps the exact process. No residual runner PID is claimed
        # from an external snapshot; any spawned-GPU-process proof is supervisor-owned.
        proc = subprocess.run([binary, *argv], input=source, text=True, capture_output=True, timeout=600)
        stdout_path.write_text(proc.stdout); stderr_path.write_text(proc.stderr)
        record = {"tag": tag, "command": [binary, *argv], "exit_code": proc.returncode, "process_reaped": True,
                  "input_path": str(input_path), "input_sha256": sha(input_path),
                  "stdout_path": str(stdout_path), "stdout_sha256": sha(stdout_path),
                  "stderr_path": str(stderr_path), "stderr_sha256": sha(stderr_path)}
        runs.append(record)
        assert proc.returncode == 0, proc.stderr[-16000:]
        rows = [json.loads(line) for line in proc.stdout.splitlines()]
        assert len(rows) == len(requests)
        if native:
            ready = [json.loads(line) for line in proc.stderr.splitlines() if line.startswith('{') and '"event":"ready"' in line]
            assert len(ready) == 1
            r = ready[0]
            assert r["backend_profile"] == PROFILE and r["flash_attention"] == "enabled"
            assert r["expected_full_attention_layers"] == 2 and r["evidence_contract_sha256"] == CONTRACT_SHA
            assert r["token_embedding_override"] == {"pattern": r"^token_embd\.weight$", "buffer_type": "CPU"}
            assert r["backend"] == args.backend and not r["causal_attention"]
            assert r["n_batch"] >= capacity and r["n_ubatch"] >= capacity
            record["ready"] = r
            for req, row in zip(requests, rows):
                if isinstance(req, dict) and req["token_ids"]:
                    validate_evidence(row, req["token_ids"], args.backend)
        return rows

    def native(tag, path, requests, capacity=256):
        return invoke(tag, runner, ["--model", str(path), "--ctx-size", str(capacity), "--threads", "2",
                      "--gpu-layers", "-1" if args.backend == "cuda" else "0"], requests, True, capacity)

    lengths = [1, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    cases = [request(f"len-{n}", [(j * 13 + 7) % 500 for j in range(n)]) for n in lengths]
    a, b = cases[4], cases[7]
    requests = cases + ["{bad}", request("empty", []), request("a-again", a["token_ids"]), b,
                        request("a-final", a["token_ids"])]
    observed = native("native", model, requests)
    assert observed[len(cases)]["error"]["code"] == "invalid_json"
    assert observed[len(cases) + 1]["error"]["code"] == "empty_prompt"
    for index in [len(cases) + 2, len(cases) + 4]:
        np.testing.assert_array_equal(observed[4]["logits"], observed[index]["logits"])
    baseline = invoke("noncausal-reference", control, [str(model), "noncausal", "256"], cases)
    causal = invoke("causal-negative", control, [str(model), "causal", "256"], [cases[0], cases[4]])
    max_error = 0.0
    # Independent non-FA graph validates all real keys and exclusion of allocated
    # padding slots at both sides of common tile boundaries. Synthetic tolerance
    # is NOT a change to the frozen production 28-case parity thresholds.
    for row, ref in zip(observed, baseline):
        delta = float(np.max(np.abs(np.asarray(row["logits"]) - ref["logits"])))
        max_error = max(max_error, delta)
        np.testing.assert_allclose(row["logits"], ref["logits"], rtol=0.02, atol=0.02)
    np.testing.assert_allclose(baseline[0]["logits"], causal[0]["logits"], rtol=1e-6, atol=1e-6)
    causal_difference = float(np.max(np.abs(np.asarray(baseline[4]["logits"]) - causal[1]["logits"])))
    assert causal_difference > 1e-4, "Synthetic fixture does not distinguish future-key visibility"
    padded = native("extra-unused-key-capacity", model, [a], capacity=512)
    # The same real keys with twice the unused cache space must not alter logits.
    np.testing.assert_allclose(observed[4]["logits"], padded[0]["logits"], rtol=2e-5, atol=2e-5)
    identity_rows = native("identity-last-raw255", identity, [request("last", [1, 7, 9]), request("single", [9]), request("other", [1, 7])])
    analytic_max = 0.0
    for row, token in zip(identity_rows, [9, 9, 7]):
        hidden = embeddings[token] / np.sqrt(np.mean(embeddings[token] ** 2) + 1e-6)
        # CPU BF16 matmul uses BF16 RHS. CUDA may accumulate the F32 activation
        # directly, so permit the bounded BF16 activation rounding difference.
        hidden = gguf.quants.dequantize(gguf.quants.quantize(hidden, gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16)
        expected = head @ hidden
        analytic_max = max(analytic_max, float(np.max(np.abs(np.asarray(row["logits"]) - expected))))
        np.testing.assert_allclose(row["logits"], expected, rtol=0.015 if args.backend == "cuda" else 2e-5,
                                   atol=0.015 if args.backend == "cuda" else 2e-5)
        assert min(row["logits"]) < 0 and abs(np.linalg.norm(row["logits"]) - 1) > 1
    np.testing.assert_array_equal(identity_rows[0]["logits"], identity_rows[1]["logits"])
    result = {"schema": "glide.native_fa_synthetic_control_result.v1", "status": "passed", "backend_profile": PROFILE,
              "backend": args.backend, "runner_sha256": sha(runner), "control_sha256": sha(control),
              "evidence_contract_sha256": CONTRACT_SHA, "fixture": "SYNTHETIC random/identity weights, NOT GLiDE quality evidence",
              "fixture_path": str(model), "fixture_sha256": sha(model), "identity_fixture_path": str(identity), "identity_fixture_sha256": sha(identity), "shape": SHAPE,
              "request_lengths": lengths, "serial_requests": requests, "runs": runs,
              "execution_evidence": [row["execution_evidence"] for row in observed if row.get("ok")],
              "max_abs_vs_nonfa_cpu": max_error, "causal_negative_max_abs": causal_difference,
              "last_raw255_analytic_max_abs": analytic_max, "aba_state_reset": "bitwise_equal",
              "padding_key_exclusion": "non-FA reference plus 256/512 capacity invariance",
              "error_recovery": ["invalid_json", "empty_prompt"], "exit_code": 0,
              "clean_process": all(run["process_reaped"] and run["exit_code"] == 0 for run in runs),
              "gpu_cleanliness": {"status": "not_observed", "requires_supervisor": True},
              "cuda_execution_proven": args.backend == "cuda"}
    (out / "control_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "backend": args.backend, "output": str(out / "control_result.json"),
                      "max_abs_vs_nonfa_cpu": max_error, "causal_negative_max_abs": causal_difference,
                      "cuda_execution_proven": args.backend == "cuda"}))

if __name__ == "__main__":
    main()
