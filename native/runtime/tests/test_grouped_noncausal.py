"""A deeper grouped-GDN fixture distinguishes noncausal from causal attention."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from tiny_fixture import create_fixture, READOUTS

runner, control = [str(Path(arg).resolve()) for arg in sys.argv[1:3]]


def invoke(binary, args, requests):
    proc = subprocess.run([binary, *args], input="".join(json.dumps(item) + "\n" for item in requests),
                          text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-15000:]
    result = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(result) == len(requests), proc.stdout
    return result


def request(identifier, tokens):
    return {"id": identifier, "token_ids": tokens, "readout_token_ids": READOUTS}


with tempfile.TemporaryDirectory() as directory:
    model = Path(directory) / "deep-grouped-SYNTHETIC.gguf"
    create_fixture(model, deep_grouped=True)
    a = [1, 8, 4, 12, 2, 19, 5, 21, 3, 30, 7, 44, 9, 60, 11, 72]
    b = [7, 6, 5, 4, 3, 2, 1, 90]
    requests = [request("a1", a), request("b", b), request("a2", a),
                request("single", [72]), request("full-batch", list(range(64))), request("a3", a)]
    native = invoke(runner, ["--model", str(model), "--ctx-size", "64", "--threads", "2"], requests)
    noncausal = invoke(control, [str(model), "noncausal"], requests)
    causal = invoke(control, [str(model), "causal"], requests)
    for i in range(len(requests)):
        assert native[i]["ok"] is True and noncausal[i]["causal"] is False and causal[i]["causal"] is True
        assert native[i]["prompt_tokens"] == len(requests[i]["token_ids"])
        assert np.isfinite(native[i]["logits"]).all()
        np.testing.assert_allclose(native[i]["logits"], noncausal[i]["logits"], rtol=0.02, atol=0.02)
    # State reset survives both a different prompt and a full-capacity physical batch.
    for repeat in [2, 5]:
        np.testing.assert_array_equal(native[0]["logits"], native[repeat]["logits"])
    # A one-token prompt has no future position to mask, providing a negative control.
    np.testing.assert_allclose(noncausal[3]["logits"], causal[3]["logits"], rtol=1e-6, atol=1e-6)
    differences = []
    for index in [0, 1, 4]:
        delta = np.abs(np.asarray(noncausal[index]["logits"]) - causal[index]["logits"])
        assert float(delta.max()) > 1e-4, "Fixture did not distinguish noncausal from causal attention"
        assert int(np.sum(delta > 1e-5)) > 200, "Difference must be broad, not a single numerical outlier"
        differences.append(float(delta.max()))
    print(json.dumps({"result": "passed", "fixture": "SYNTHETIC, NOT GLiDE", "layer_types": ["recurrent", "full", "recurrent", "full"],
        "key_heads": 2, "value_heads": 6, "value_to_key_ratio": 3, "native_matches_noncausal_control": "allclose_rtol_atol_0.02",
        "aba_state_reset": "bitwise_equal", "single_token_control": "equal_within_1e-6",
        "causal_vs_noncausal_max_abs_differences": differences, "whole_prompt_lengths": [len(item["token_ids"]) for item in requests],
        "gpu_tested": False}))
