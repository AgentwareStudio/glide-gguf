"""Source verification unit tests mutate only disposable miniature trees."""
import hashlib
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_llama_source import verify_tree


def must_reject(root, manifest, needle):
    try:
        verify_tree(root, manifest)
    except ValueError as exc:
        assert needle in str(exc), str(exc)
    else:
        raise AssertionError("Modified source unexpectedly accepted")


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "src").mkdir()
    source = root / "src/model.cpp"
    original = b"int model() { return 1; }\n"
    source.write_bytes(original)
    manifest = {"src/model.cpp": hashlib.sha256(original).hexdigest()}
    (root / "PINNED_REVISION").write_text("8e5a58df87ab59dfceb5b961ae26691e39f30c24\n")
    verify_tree(root, manifest)
    source.write_bytes(original + b"// injected edit with unchanged revision marker\n")
    must_reject(root, manifest, "content changed")
    source.write_bytes(original)
    extra = root / "src/injected.cpp"
    extra.write_text("int injected = 1;\n")
    must_reject(root, manifest, "additional build input")
    extra.unlink()
    source.unlink()
    must_reject(root, manifest, "missing")
    outside = root / "outside.txt"
    outside.write_bytes(original)
    source.symlink_to(outside)
    must_reject(root, manifest, "redirected")
print("5 source-tree verification scenarios passed, including forged matching revision marker")
