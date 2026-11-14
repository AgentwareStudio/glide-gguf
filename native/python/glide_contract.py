"""Fail-closed identity and protocol checks for the private GLiDE GGUF build."""
from __future__ import annotations

import ast
import hashlib
import types
import json
import math
import os
from pathlib import Path

LOCK_PATH = Path(__file__).with_name('source_lock.json')
SOURCE_LOCK = json.loads(LOCK_PATH.read_text())
REVISION = SOURCE_LOCK['revision']
LLAMA_COMMIT = SOURCE_LOCK['llama_cpp_commit']


def strict_json(data):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_child(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError(f'Path escapes source directory: {relative}')
    return p


def verify_file(root, relative, expected):
    p = safe_child(root, relative)
    if not p.is_file() or p.stat().st_size != expected['bytes']:
        raise ValueError(f'Missing or wrong-size source file: {relative}')
    before = p.stat()
    if sha256_file(p) != expected['sha256']:
        raise ValueError(f'SHA256 mismatch: {relative}')
    after = p.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed while hashing: {relative}')


def validate_source_files(root):
    root = Path(root)
    for name, expected in SOURCE_LOCK['source_files'].items():
        verify_file(root, name, expected)
    config = json.loads((root / 'config.json').read_text())
    decision = json.loads((root / 'decision_config.json').read_text())
    validate_decision_config(decision)
    t = config['text_config']
    expected = {'hidden_size': 5120, 'vocab_size': 248320, 'num_hidden_layers': 64,
                'num_attention_heads': 24, 'num_key_value_heads': 4}
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('GLiDE text architecture differs from pinned source')
    if t['layer_types'] != (['linear_attention'] * 3 + ['full_attention']) * 16:
        raise ValueError('Unexpected hybrid attention layout')
    return config, decision


def validate_decision_config(d):
    if d.get('format_version') != 1 or d.get('prompt_format') != 'glide-v1':
        raise ValueError('Not a supported GLiDE decision configuration')
    if d.get('attention_mode') != 'noncausal_full_attention' or d.get('pooling') != 'last':
        raise ValueError('GLiDE attention/pooling semantic mismatch')
    if type(d.get('temperature')) not in (int, float) or d['temperature'] != 1.0:
        raise ValueError('Unsupported GLiDE temperature')
    ids, codes = d.get('token_ids'), d.get('codes')
    if not isinstance(ids, list) or not isinstance(codes, list) or len(ids) != 255 or len(codes) != 255:
        raise ValueError('Require all 255 ordered decision code rows')
    if any(type(i) is not int or not 0 <= i < 248320 for i in ids) or len(set(ids)) != 255:
        raise ValueError('Invalid or duplicate decision token ids')
    if any(not isinstance(c, str) or not c for c in codes) or len(set(codes)) != 255:
        raise ValueError('Invalid or duplicate decision codes')


def verified_source_bytes(root, relative):
    """Return the exact authenticated bytes, never a timestamp-trusted pyc."""
    raw = safe_child(root, relative).read_bytes()
    expected = SOURCE_LOCK['source_files'][relative]
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise ValueError(f'SHA256/size mismatch: {relative}')
    return raw


def load_prompt_module(root):
    raw = verified_source_bytes(root, 'glide/prompt.py')
    p = safe_child(root, 'glide/prompt.py')
    module = types.ModuleType('_pinned_glide_prompt')
    module.__file__ = str(p)
    exec(compile(raw, str(p), 'exec'), module.__dict__)
    return module


def load_official_postprocessing(root):
    """Load only two pure functions from the hash-pinned upstream source.

    This avoids importing CUDA/image/model dependencies just to reproduce the
    official softmax and response semantics. No arbitrary remote code loading.
    """
    raw = verified_source_bytes(root, 'glide/model.py')
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_softmax', '_answer'}]
    if [n.name for n in nodes] != ['_softmax', '_answer']:
        raise ValueError('Pinned official postprocessing functions missing')
    env = {'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-glide-postprocessing>', 'exec'), env)
    return env['_softmax'], env['_answer']


def canonical_ids(ids):
    return json.dumps(ids, separators=(',', ':'))


def write_json(path, value):
    p = Path(path)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_new_json(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def commit_new_file(staging, destination):
    """Same-directory hard-link commit is atomic and never replaces a target."""
    os.link(staging, destination)
    Path(staging).unlink()
