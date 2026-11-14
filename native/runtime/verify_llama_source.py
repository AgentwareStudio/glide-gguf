#!/usr/bin/env python3
"""Fail closed if pinned official llama.cpp source content has changed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REVISION = "8e5a58df87ab59dfceb5b961ae26691e39f30c24"
ARCHIVE_SHA256 = "f1205ecd56f64fdc012e0b5bcd38defa7aaccec3d82aaaed45001d7bd6e30de0"
MANIFEST_SHA256 = "d46f476654a4c50de0ce49f47dc8e283daa07991e709b01345d530249b40560d"
NATIVE_ROOTS = {"src", "include", "ggml", "vendor", "cmake", "common"}
BUILD_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".m", ".mm", ".s", ".asm", ".cmake", ".in", ".inc", ".def", ".metal", ".comp"}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tree(root, files):
    """Check all original bytes and reject extra potentially compiled build inputs."""
    root = root.resolve()
    for relative, expected in files.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Pinned source missing, redirected, or not a regular file: {relative}")
        if digest_file(path) != expected:
            raise ValueError(f"Pinned source content changed: {relative}")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        name = relative.as_posix()
        relevant = (relative.parts[0] in NATIVE_ROOTS or path.name == "CMakeLists.txt" or
                    path.suffix.lower() in BUILD_SUFFIXES)
        if name not in files and relevant and (path.is_file() or path.is_symlink()):
            raise ValueError(f"Unpinned additional build input: {name}")


def verify(root):
    manifest_path = Path(__file__).with_name("llama_source_manifest.json")
    if digest_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Pinned source manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != REVISION or manifest["archive_sha256"] != ARCHIVE_SHA256 or manifest.get("links"):
        raise ValueError("Unsupported source manifest provenance")
    verify_tree(root, manifest["files"])
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    try:
        count = verify(args.source)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Source verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {count} official source files at {REVISION}; no extra native build inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
