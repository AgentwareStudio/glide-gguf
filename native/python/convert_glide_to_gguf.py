"""Explicit GLiDE -> standard Qwen35 GGUF with compact classifier readout.

This deliberately bypasses stock model autodetection, whose Pplx matcher also
matches GLiDE. The result requires the supplied GLiDE runner, NOT a chat UI or
the stock D3/Pplx System One endpoint. No weights are downloaded by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import struct
import sys

from glide_contract import (LLAMA_COMMIT, REVISION, SOURCE_LOCK, canonical_ids,
                            commit_new_file, safe_child, sha256_file, strict_json,
                            validate_source_files, verify_file, write_new_json)

TENSOR_CONTRACT = json.loads(Path(__file__).with_name('tensor_contract.json').read_text())


def verify_effective_inventory(root, manifest):
    files = manifest.get('output_files', {})
    records = manifest.get('tensors', {})
    index_path = safe_child(root, 'model.safetensors.index.json')
    if 'model.safetensors.index.json' not in files or not index_path.is_file():
        raise ValueError('Effective-weight index is not hash-bound')
    index = strict_json(index_path.read_text())
    weight_map = index.get('weight_map', {})
    if set(records) != set(TENSOR_CONTRACT) or set(weight_map) != set(TENSOR_CONTRACT):
        raise ValueError('Effective backbone inventory must contain exactly the pinned 850 tensors')
    shards = set(weight_map.values())
    if not shards <= set(files):
        raise ValueError('Effective shard index references an unhashed file')
    observed = {}
    for name in shards:
        path = safe_child(root, name)
        if not name.startswith('model-') or not name.endswith('.safetensors'):
            raise ValueError('Unsupported effective shard filename')
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if length > 16 << 20 or length + 8 > path.stat().st_size:
                raise ValueError('Invalid safetensors header length')
            header = strict_json(f.read(length))
        payload_bytes = path.stat().st_size - length - 8
        ranges = []
        for tensor, spec in header.items():
            if tensor == '__metadata__':
                continue
            expected = TENSOR_CONTRACT.get(tensor)
            if tensor in observed or expected is None or weight_map.get(tensor) != name:
                raise ValueError('Duplicate, unexpected, or misplaced effective tensor')
            offsets = spec.get('data_offsets', [])
            if (spec.get('dtype') != 'BF16' or spec.get('shape') != expected['shape'] or
                    any(type(x) is not int for x in spec.get('shape', [])) or
                    len(offsets) != 2 or any(type(x) is not int for x in offsets) or
                    not 0 <= offsets[0] < offsets[1] <= payload_bytes or
                    offsets[1] - offsets[0] != expected['bytes']):
                raise ValueError(f'Effective tensor shape/type/size mismatch: {tensor}')
            ranges.append(tuple(offsets))
            rec = records[tensor]
            if any(rec.get(k) != expected[k] for k in ('shape', 'dtype', 'bytes')) or rec.get('file') != name:
                raise ValueError(f'Effective tensor record mismatch: {tensor}')
            observed[tensor] = name
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError('Effective safetensors payload has holes or overlaps')
            cursor = end
        if cursor != payload_bytes:
            raise ValueError('Effective safetensors payload has unaccounted or missing bytes')
    if observed != weight_map:
        raise ValueError('Effective shard/index coverage differs')


def verify_weight_source(root, kind):
    root = Path(root)
    if kind == 'original':
        for name, expected in SOURCE_LOCK['original_weights'].items():
            verify_file(root, name, expected)
        index = json.loads((root / 'model.safetensors.index.json').read_text())
        canonical = json.dumps(index, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != '8898248e2822192f5bca2743d0b82e02c254d81cea41631123c057b17e165377':
            raise ValueError('Original shard index mismatch')
        return {'kind': 'original-bf16', 'revision': REVISION,
                'weight_files': SOURCE_LOCK['original_weights']}
    manifest_path = root / 'effective_weights_manifest.json'
    m = strict_json(manifest_path.read_text())
    if m.get('schema') != 'glide.effective_weights.v1' or m.get('source_revision') != REVISION:
        raise ValueError('Unsupported effective-weight intermediate')
    if m.get('source_packed_manifest_sha256') != SOURCE_LOCK['packed_manifest_sha256']:
        raise ValueError('Effective weights are not from the selected frozen packed model')
    # The exporter commits a complete hash manifest only after all tensors exist.
    if m.get('status') != 'complete':
        raise ValueError('Effective-weight export is not committed complete')
    files = m.get('output_files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Effective-weight output file inventory is absent')
    for name, record in files.items():
        verify_file(root, name, record)
    verify_effective_inventory(root, m)
    return {'kind': 'reconstructed-packed-effective-bf16', 'revision': REVISION,
            'effective_manifest_sha256': sha256_file(manifest_path),
            'packed_manifest_sha256': SOURCE_LOCK['packed_manifest_sha256']}


def build_model_class(llama_root):
    llama_root = Path(llama_root).resolve()
    lock = json.loads(Path(__file__).with_name('llama_source_lock.json').read_text())
    if lock['commit'] != LLAMA_COMMIT:
        raise ValueError('Unexpected llama.cpp source lock')
    for relative, expected in lock['files'].items():
        if sha256_file(safe_child(llama_root, relative)) != expected:
            raise ValueError(f'Pinned llama.cpp Python source differs: {relative}')
    for name, module in list(sys.modules.items()):
        if name in {'conversion', 'gguf'} or name.startswith(('conversion.', 'gguf.')):
            origin = getattr(module, '__file__', None)
            allowed = llama_root / ('gguf-py' if name == 'gguf' or name.startswith('gguf.') else 'conversion')
            if origin is None or not Path(origin).resolve().is_relative_to(allowed):
                raise ValueError(f'Cached module comes from a different llama.cpp checkout: {name}')
    sys.path.insert(0, str(llama_root))
    sys.path.insert(0, str(llama_root / 'gguf-py'))
    import torch
    import gguf
    from conversion.qwen import Qwen3_5TextModel

    class GlideGGUFModel(Qwen3_5TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35
        no_mtp = True

        def __init__(self, *args, glide_decision, glide_provenance, **kwargs):
            self.glide_decision = glide_decision
            self.glide_provenance = glide_provenance
            if 'hparams' not in kwargs:
                raise ValueError('Explicit verified hparams required; never autodetect GLiDE')
            super().__init__(*args, **kwargs)

        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            w = self.gguf_writer
            w.add_causal_attention(False)
            w.add_embedding_length_out(255)
            w.add_pooling_type(gguf.PoolingType.LAST)
            w.add_uint32('glide.format_version', 1)
            w.add_string('glide.attention_mode', 'noncausal_full_attention')
            w.add_string('glide.prompt_format', 'glide-v1')
            w.add_string('glide.pooling', 'last')
            w.add_string('glide.source_revision', REVISION)
            w.add_bool('glide.text_only', True)
            w.add_string('glide.readout_layout', 'cls.output')
            w.add_string('glide.readout_dtype', 'BF16')
            w.add_string('glide.readout_token_ids_json', canonical_ids(self.glide_decision['token_ids']))
            w.add_string('glide.tokenizer_sha256', SOURCE_LOCK['source_files']['tokenizer.json']['sha256'])
            w.add_string('glide.decision_config_sha256', SOURCE_LOCK['source_files']['decision_config.json']['sha256'])
            w.add_string('glide.llama_cpp_commit', LLAMA_COMMIT)
            w.add_string('glide.weight_source', self.glide_provenance['kind'])
            w.add_string('glide.provenance_json', json.dumps(self.glide_provenance, sort_keys=True, separators=(',', ':')))

        def generate_extra_tensors(self):
            from safetensors.torch import load_file
            data = load_file(self.dir_model / 'readout.safetensors')
            if set(data) != {'weight'} or tuple(data['weight'].shape) != (255, 5120) or data['weight'].dtype != torch.bfloat16:
                raise ValueError('Pinned GLiDE readout must be BF16 [255,5120]')
            yield 'glide_readout.weight', data['weight']

        def modify_tensors(self, data_torch, name, bid):
            if name == 'glide_readout.weight':
                yield 'cls.output.weight', data_torch
                return
            # The pinned upstream filter strips language_model. before this
            # method, leaving model.layers.*, model.norm.*, model.embed_tokens.*.
            if not name.startswith('model.'):
                raise ValueError(f'Unexpected text tensor: {name}')
            yield from super().modify_tensors(data_torch, name, bid)

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if new_name == 'cls.output.weight':
                return gguf.GGMLQuantizationType.BF16
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    return GlideGGUFModel, gguf, torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--source-kind', choices=['original', 'effective-packed'], required=True)
    p.add_argument('--llama-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    config, decision = validate_source_files(args.source_dir)
    provenance = verify_weight_source(args.source_dir, args.source_kind)
    # The exact imported llama.cpp Python sources are also checked below.
    receipt_path = args.output.with_suffix(args.output.suffix + '.receipt.json')
    if any(p.exists() or p.is_symlink() for p in [args.output, receipt_path]):
        raise FileExistsError('Refusing to overwrite existing output or receipt')
    if args.verify_only:
        print(json.dumps({'status': 'source_verified', 'provenance': provenance}))
        return
    model_class, gguf, torch = build_model_class(args.llama_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.incomplete')
    if tmp.exists() or tmp.is_symlink():
        raise FileExistsError(tmp)
    logging.basicConfig(level=logging.INFO)
    with torch.inference_mode():
        model = model_class(args.source_dir, gguf.LlamaFileType.MOSTLY_BF16, tmp,
                            hparams=config, glide_decision=decision,
                            glide_provenance=provenance, use_temp_file=True,
                            model_name='GLiDE text-only decision reference')
        normalized = {name.replace('language_model.', '') for name in TENSOR_CONTRACT}
        if set(model.model_tensors) != normalized:
            raise ValueError('Converter indexed an incomplete or unexpected backbone tensor inventory')
        model.write()
    receipt = {'schema': 'glide.gguf_conversion.v1', 'status': 'written_not_inference_validated',
               'source': provenance, 'llama_cpp_commit': LLAMA_COMMIT,
               'bytes': tmp.stat().st_size, 'sha256': sha256_file(tmp),
               'readout': 'BF16 [255,5120] in cls.output.weight; raw last pooled projection',
               'runtime': 'Dedicated GLiDE runner required; not a generic chat or D3/Pplx model',
               'limitations': ['No numerical parity claim before real-model tests',
                               'No quality, memory or hardware performance inherited from packed NF4']}
    receipt_tmp = receipt_path.with_name(receipt_path.name + '.incomplete')
    write_new_json(receipt_tmp, receipt)
    commit_new_file(tmp, args.output)
    commit_new_file(receipt_tmp, receipt_path)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
