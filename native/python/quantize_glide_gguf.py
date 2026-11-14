"""Explicit native GGUF quantization with protected compact decision readout.

No downloaded weights, GPU allocation, score-based recipe tuning or publication.
Input must be the original-BF16 reference built by this adapter. Output remains
unvalidated until separate real-model parity/quality/memory gates complete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from glide_contract import (LLAMA_COMMIT, REVISION, SOURCE_LOCK, commit_new_file,
                            sha256_file, strict_json, write_new_json)
from convert_glide_to_gguf import TENSOR_CONTRACT, build_model_class

PROTECTION_SHA256 = '09aad5382a11683203fe83fe113a6584fc4145fd6659fac73626c6feabf3a0e8'
ALLOWED_METADATA_CHANGES = {'general.file_type', 'general.quantization_version', 'GGUF.kv_count'}


def protected_names(policy_file):
    policy_bytes = Path(policy_file).read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != PROTECTION_SHA256:
        raise ValueError('Protection file differs from the frozen 17-tensor policy')
    names = []
    for line in policy_bytes.decode('utf-8').splitlines():
        if not line or not line.startswith('^') or not line.endswith('$=bf16'):
            raise ValueError('Protection file must contain anchored exact BF16 tensor rules only')
        raw = line[1:-6]
        name = raw.replace('\\.', '.')
        if '^' + re.escape(name) + '$=bf16' != line:
            raise ValueError('Only exact tensor names are allowed, not arbitrary regex rules')
        names.append(name)
    if len(names) != 17 or len(names) != len(set(names)) or names[0] != 'cls.output.weight':
        raise ValueError('Protection must contain the frozen head and 16 restorations')
    return names


def expected_inventory(gguf):
    """Pinned text-only conversion shapes, in GGUF dimension order."""
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 64)
    result = {'cls.output.weight': [5120, 255]}
    for original, spec in TENSOR_CONTRACT.items():
        name = original.replace('language_model.', '')
        if name.endswith('.dt_bias'):
            name = name[:-8] + '.dt_proj.bias'
        mapped = mapping.get_name(name, try_suffixes=('.weight', '.bias'))
        if mapped is None or mapped in result:
            raise ValueError(f'Invalid or duplicate pinned tensor mapping: {original}')
        shape = [n for n in spec['shape'] if n != 1] if '.conv1d.' in name else spec['shape']
        result[mapped] = list(reversed(shape))
    if len(result) != 851:
        raise ValueError('Pinned GGUF inventory must contain 850 backbone tensors and one readout')
    return result


def expected_source_type(name, shape, gguf):
    if len(shape) <= 1 or name.endswith('_norm.weight') or 'ssm_conv1d' in name:
        return gguf.GGMLQuantizationType.F32
    return gguf.GGMLQuantizationType.BF16


def validate_reference_source(source, gguf):
    """Require the complete, hash-bound original conversion, not identity strings alone."""
    source = Path(source)
    receipt_path = source.with_suffix(source.suffix + '.receipt.json')
    receipt = strict_json(receipt_path.read_text())
    provenance = {'kind': 'original-bf16', 'revision': REVISION,
                  'weight_files': SOURCE_LOCK['original_weights']}
    if (receipt.get('schema') != 'glide.gguf_conversion.v1' or
            receipt.get('status') != 'written_not_inference_validated' or
            receipt.get('llama_cpp_commit') != LLAMA_COMMIT or receipt.get('source') != provenance or
            type(receipt.get('bytes')) is not int or receipt['bytes'] != source.stat().st_size):
        raise ValueError('Original-BF16 conversion receipt identity is absent or inconsistent')
    digest = sha256_file(source)
    if receipt.get('sha256') != digest:
        raise ValueError('Source GGUF differs from its original conversion receipt')
    reader = gguf.GGUFReader(str(source))
    tensors = {t.name: t for t in reader.tensors}
    expected = expected_inventory(gguf)
    if len(tensors) != len(reader.tensors) or set(tensors) != set(expected):
        raise ValueError('Source GGUF does not contain the exact 851 pinned tensors')
    for name, shape in expected.items():
        if list(tensors[name].shape) != shape or tensors[name].tensor_type != expected_source_type(name, shape, gguf):
            raise ValueError(f'Source GGUF shape/dtype differs from the pinned conversion: {name}')
    required = {
        'general.architecture': 'qwen35', 'general.file_type': int(gguf.LlamaFileType.MOSTLY_BF16),
        'glide.weight_source': 'original-bf16', 'glide.source_revision': REVISION,
        'glide.llama_cpp_commit': LLAMA_COMMIT, 'glide.format_version': 1,
        'glide.attention_mode': 'noncausal_full_attention', 'glide.prompt_format': 'glide-v1',
        'glide.pooling': 'last', 'glide.text_only': True,
        'glide.readout_layout': 'cls.output', 'glide.readout_dtype': 'BF16',
        'glide.tokenizer_sha256': SOURCE_LOCK['source_files']['tokenizer.json']['sha256'],
        'glide.decision_config_sha256': SOURCE_LOCK['source_files']['decision_config.json']['sha256'],
        'qwen35.attention.causal': False, 'qwen35.pooling_type': int(gguf.PoolingType.LAST),
        'qwen35.embedding_length_out': 255, 'qwen35.embedding_length': 5120,
        'qwen35.block_count': 64,
    }
    for name, value in required.items():
        field = reader.get_field(name)
        if field is None or type(field.contents()) is not type(value) or field.contents() != value:
            raise ValueError(f'Source GGUF metadata differs from the pinned conversion: {name}')
    provenance_field = reader.get_field('glide.provenance_json')
    if provenance_field is None or strict_json(provenance_field.contents()) != provenance:
        raise ValueError('Source GGUF provenance and conversion receipt differ')
    vocabulary = reader.get_field('tokenizer.ggml.tokens')
    if vocabulary is None or len(vocabulary.contents()) != 248320:
        raise ValueError('Source GGUF vocabulary has an unexpected size')
    return digest, receipt_path


def expected_recipe_types(tensors, protected, recipe, gguf, *, n_layers=64):
    """Mirror the pinned Qwen35-only K-quant mixture and shape fallback rules.

    Counters deliberately skip manual protected overrides, as the pinned native
    implementation does. This is not a generic quantizer policy implementation.
    """
    if recipe not in {'Q4_K_S', 'Q4_K_M', 'Q5_K_M'}:
        raise ValueError('Unsupported frozen native quantization recipe')
    q = gguf.GGMLQuantizationType
    base = q.Q5_K if recipe == 'Q5_K_M' else q.Q4_K
    def v_like(name):
        return any(s in name for s in ('attn_v.weight', 'attn_qkv.weight', 'attn_kv_b.weight'))
    n_v = sum(v_like(t.name) for t in tensors)
    i_v = i_down = 0
    def key(t):
        match = re.match(r'blk\.(\d+)\.', t.name)
        return (int(match[1]) if match else -1, t.name)
    def more_bits(i, n):
        return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2
    result = {}
    for tensor in sorted(tensors, key=key):
        name, shape = tensor.name, list(tensor.shape)
        if (len(shape) < 2 or not name.endswith('weight') or
                '_norm.weight' in name or 'ssm_conv1d' in name):
            target = tensor.tensor_type
        elif name == 'token_embd.weight':
            target = q.Q8_0
        elif name in protected:
            target = q.BF16
        else:
            target = base
            if v_like(name):
                if recipe == 'Q4_K_S' and i_v < 4:
                    target = q.Q5_K
                elif recipe in {'Q4_K_M', 'Q5_K_M'} and more_bits(i_v, n_v):
                    target = q.Q6_K
                i_v += 1
            elif 'ffn_down' in name:
                if recipe == 'Q4_K_S' and i_down < n_layers // 8:
                    target = q.Q5_K
                elif recipe in {'Q4_K_M', 'Q5_K_M'} and more_bits(i_down, n_layers):
                    target = q.Q6_K
                i_down += 1
        block = gguf.GGML_QUANT_SIZES[target][0]
        if shape[0] % block:
            target = {q.Q4_K: q.Q4_0, q.Q5_K: q.Q5_1, q.Q6_K: q.Q8_0}.get(target, target)
            if shape[0] % gguf.GGML_QUANT_SIZES[target][0]:
                target = q.F16
        result[name] = target
    return result


def tensor_digest(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()


def verify_quantized(source, output, protected, gguf, *, recipe=None):
    before = gguf.GGUFReader(str(source))
    after = gguf.GGUFReader(str(output))
    old = {t.name: t for t in before.tensors}
    new = {t.name: t for t in after.tensors}
    if (len(old) != len(before.tensors) or len(new) != len(after.tensors) or
            set(old) != set(new) or 'output.weight' in new or 'cls.output.weight' not in new):
        raise ValueError('Native quantizer changed backbone/readout tensor inventory')
    for name in old:
        if list(old[name].shape) != list(new[name].shape):
            raise ValueError(f'Native quantizer changed tensor shape: {name}')
    before_keys = set(before.fields) - ALLOWED_METADATA_CHANGES
    after_keys = set(after.fields) - ALLOWED_METADATA_CHANGES
    if before_keys != after_keys:
        raise ValueError('Native quantizer added or removed unapproved metadata')
    for key in before_keys:
        field, changed = before.fields[key], after.fields[key]
        if field.types != changed.types or field.contents() != changed.contents():
            raise ValueError(f'Native quantizer changed model metadata: {key}')
    if recipe is not None:
        layer_field = before.get_field('qwen35.block_count')
        expected = expected_recipe_types(before.tensors, protected, recipe, gguf,
                                         n_layers=layer_field.contents() if layer_field else 64)
        for name, target_type in expected.items():
            if new[name].tensor_type != target_type:
                raise ValueError(f'Native quantizer used an unexpected tensor type: {name}')
        if 'token_embd.weight' not in new or new['token_embd.weight'].tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise ValueError('The native recipe requires a Q8_0 token embedding')
        for key, expected_value in [('general.file_type', int(getattr(gguf.LlamaFileType, 'MOSTLY_' + recipe))),
                                    ('general.quantization_version', gguf.GGML_QUANT_VERSION)]:
            field = after.get_field(key)
            if field is None or field.types != [gguf.GGUFValueType.UINT32] or field.contents() != expected_value:
                raise ValueError(f'Native quantizer output metadata differs from the recipe: {key}')
    preserved = {}
    for name in protected:
        if name not in old or old[name].tensor_type != gguf.GGMLQuantizationType.BF16 or new[name].tensor_type != gguf.GGMLQuantizationType.BF16:
            raise ValueError(f'Protected tensor was missing or quantized: {name}')
        digest = tensor_digest(old[name])
        if tensor_digest(new[name]) != digest:
            raise ValueError(f'Protected BF16 bytes changed: {name}')
        preserved[name] = digest
    if not any(t.tensor_type not in {gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16} for t in after.tensors):
        raise ValueError('No native quantized tensor found')
    counts = {}
    for tensor in after.tensors:
        counts[tensor.tensor_type.name] = counts.get(tensor.tensor_type.name, 0) + 1
    return {'protected_bf16_sha256': preserved, 'tensor_type_counts': counts, 'tensor_count': len(new)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--llama-root', type=Path, required=True)
    p.add_argument('--quantizer', type=Path, required=True)
    p.add_argument('--recipe', choices=['Q4_K_S', 'Q4_K_M', 'Q5_K_M'], required=True)
    p.add_argument('--protected-tensors', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--timeout-seconds', type=float, default=3600)
    args = p.parse_args()
    if not 1 <= args.threads <= 1024 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError('threads must be 1..1024 and timeout must be positive and finite')
    _, gguf, _ = build_model_class(args.llama_root)
    protected = protected_names(args.protected_tensors)
    source_hash, source_receipt = validate_reference_source(args.source, gguf)
    source_receipt_hash = sha256_file(source_receipt)
    quantizer_hash = sha256_file(args.quantizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.incomplete')
    receipt_path = args.output.with_suffix(args.output.suffix + '.receipt.json')
    if any(p.exists() or p.is_symlink() for p in [args.output, tmp, receipt_path]):
        raise FileExistsError('Refusing to replace an output, staging file, or receipt')
    command = [str(args.quantizer.resolve()), '--tensor-type-file', str(args.protected_tensors.resolve()),
               '--token-embedding-type', 'q8_0', str(args.source.resolve()), str(tmp.resolve()),
               args.recipe, str(args.threads)]
    subprocess.run(command, check=True, timeout=args.timeout_seconds)
    checks = verify_quantized(args.source, tmp, protected, gguf, recipe=args.recipe)
    if sha256_file(args.source) != source_hash:
        raise ValueError('Source reference changed during quantization')
    if (sha256_file(source_receipt) != source_receipt_hash or
            sha256_file(args.quantizer) != quantizer_hash or
            sha256_file(args.protected_tensors) != PROTECTION_SHA256):
        raise ValueError('Source receipt, quantizer or protection policy changed during quantization')
    receipt = {'schema': 'glide.native_gguf_quantization.v1', 'status': 'written_not_quality_validated',
               'source_sha256': source_hash, 'source_kind': 'direct_original_bf16',
               'bytes': tmp.stat().st_size, 'sha256': sha256_file(tmp),
               'recipe': args.recipe, 'token_embedding_type': 'Q8_0',
               'command': command, 'quantizer_sha256': quantizer_hash,
               'source_conversion_receipt_sha256': source_receipt_hash,
               'protection_file_sha256': PROTECTION_SHA256, **checks,
               'warnings': ['This is not the prior bnb NF4 artifact.',
                            'The 285 exact embedding-row overrides of the old row-INT8 package are not retained by native Q8_0.',
                            'Transferred BF16 restoration choices are not new GGUF calibration results.',
                            'Actual model quality and 16 GiB deployment memory are not yet validated.']}
    receipt_tmp = receipt_path.with_name(receipt_path.name + '.incomplete')
    write_new_json(receipt_tmp, receipt)
    commit_new_file(tmp, args.output)
    commit_new_file(receipt_tmp, receipt_path)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
