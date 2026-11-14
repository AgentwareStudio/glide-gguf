#!/usr/bin/env python3
"""Download and hash-check the exact official llama.cpp source used by this runtime."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile
import urllib.request

REVISION = "8e5a58df87ab59dfceb5b961ae26691e39f30c24"
SHA256 = "f1205ecd56f64fdc012e0b5bcd38defa7aaccec3d82aaaed45001d7bd6e30de0"
URL = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{REVISION}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent / "llama.cpp")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error(f"Destination already exists; will not replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="glide-source-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "source.tar.gz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("Official source archive SHA-256 mismatch; refusing extraction")
        with tarfile.open(archive) as source:
            # Python 3.12+ data filter rejects traversal, external links, and device files.
            source.extractall(temporary, filter="data")
        extracted = temporary / f"llama.cpp-{REVISION}"
        (extracted / "PINNED_REVISION").write_text(REVISION + "\n")
        extracted.rename(destination)
    print(f"Verified official source: {destination}\nRevision: {REVISION}\nArchive SHA-256: {SHA256}")


if __name__ == "__main__":
    main()
