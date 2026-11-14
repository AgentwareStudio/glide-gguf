"""Tests that do not require model weights. Every process has an outer timeout."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = sys.argv[1]

def invoke(*args):
    return subprocess.run([binary, *args], text=True, capture_output=True, timeout=10)

for args in [[], ["--bad"], ["--model"], ["--model", "x", "--ctx-size", "0"],
             ["--model", "x", "--ctx-size", "1.2"], ["--model", "x", "--threads", "-1"],
             ["--model", "x", "--timeout-ms", "0"], ["--model", "x", "--model", "y"]]:
    result = invoke(*args)
    assert result.returncode == 2 and not result.stdout, (args, result)
    assert json.loads(result.stderr.splitlines()[-1])["code"] == "invalid_argument"
assert invoke("--help").returncode == 0
assert "8e5a58df87ab59dfceb5b961ae26691e39f30c24" in invoke("--version").stdout
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "bad.gguf"
    path.write_bytes(b"not a GGUF file")
    result = invoke("--model", str(path))
    assert result.returncode == 2 and not result.stdout
    assert json.loads(result.stderr.splitlines()[-1])["code"] == "model_open_failed"
print("11 CLI scenarios passed")
