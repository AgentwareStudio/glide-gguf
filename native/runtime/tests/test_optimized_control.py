"""Even optimized interpreters must not weaken the CUDA synthetic admission tool."""
from pathlib import Path
import subprocess
import sys
script = Path(__file__).with_name("test_fa_production_control.py")
result = subprocess.run([sys.executable, "-O", str(script), "--help"], text=True, capture_output=True, timeout=30)
if result.returncode == 0 or "refuses optimized Python" not in result.stderr:
    raise RuntimeError("Optimized admission control failed to reject before any work")
print("Optimized control rejects before arguments, fixture generation, or runner launch")
