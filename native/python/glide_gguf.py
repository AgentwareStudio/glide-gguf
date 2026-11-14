"""Isolated FA+host-embedding GLiDE API with observed execution evidence.

Official prompt, option order, softmax and response functions are loaded only
from hash-pinned local sources. Each question is an independent whole-prompt
forward. There is no generation, generic chat protocol, or multimodal support.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import struct
import subprocess
import threading
import time

from glide_contract import (canonical_ids, load_official_postprocessing,
                            load_prompt_module, strict_json, validate_source_files)

BACKEND_PROFILE = 'glide-fa-hostemb-v1'
EVIDENCE_CONTRACT_SHA256 = 'e9b7c3b70d9cd5af8ce776ffa8e955f1c23d9cef50c84b66a14fd72ae599a433'


def verify_evidence_contract():
    path = Path(__file__).resolve().parent.parent / 'EXECUTION_EVIDENCE_CONTRACT.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != EVIDENCE_CONTRACT_SHA256:
        raise ValueError('FA execution evidence contract differs from the reviewed version')


def validate_execution_evidence(evidence, *, prompt_tokens, expected_layers=16, backend='cuda'):
    """Validate observed execution, never promote requested policy to evidence.

    This is a transport/shape check. Executable/source/build authentication and
    measured allocation provenance remain the caller's separate responsibility.
    """
    def need(condition, message):
        if not condition:
            raise RuntimeError('Invalid FA execution evidence: ' + message)
    need(type(prompt_tokens) is int and prompt_tokens > 0, 'invalid expected token count')
    need(type(expected_layers) is int and expected_layers > 0, 'invalid expected layer count')
    need(backend in ('cpu', 'cuda'), 'unsupported expected backend')
    keys = {'schema', 'backend_profile', 'requested_backend', 'decode_calls', 'prompt_tokens',
            'expected_full_attention_layers', 'state_reset_before', 'state_reset_after',
            'flash_attention_nodes', 'classifier_nodes', 'token_embedding_nodes',
            'vocabulary_matmul_nodes'}
    need(type(evidence) is dict and set(evidence) == keys, 'missing or unexpected fields')
    need(evidence['schema'] == 'glide.native_execution_evidence.v1' and
         evidence['backend_profile'] == BACKEND_PROFILE, 'wrong profile/schema')
    need(evidence['requested_backend'] == backend, 'requested backend differs')
    for name, expected in [('decode_calls', 1), ('prompt_tokens', prompt_tokens),
                           ('expected_full_attention_layers', expected_layers)]:
        need(type(evidence[name]) is int and evidence[name] == expected, name + ' differs')
    need(evidence['state_reset_before'] is True and evidence['state_reset_after'] is True,
         'state reset not observed')

    def device(value, expected):
        need(type(value) is dict and set(value) == {'backend', 'type', 'name', 'buffer'},
             'malformed device record')
        need(all(type(v) is str and 0 < len(v) <= 256 for v in value.values()),
             'invalid device description')
        if expected == 'cuda':
            need(value['backend'] == 'CUDA' and value['type'] in ('GPU', 'IGPU'),
                 'required CUDA node/weight not on CUDA')
        elif expected == 'cpu':
            need(value['backend'] == 'CPU' and value['type'] == 'CPU',
                 'required host weight/node not on CPU')
        else:
            need((value['backend'] == 'CPU' and value['type'] == 'CPU') or
                 (value['backend'] == 'CUDA' and value['type'] in ('GPU', 'IGPU')),
                 'unexpected lookup device')

    def node(value, op, weight=None):
        fields = {'name', 'op', 'output_device', 'phase', 'executed'}
        if weight is not None:
            fields |= {'weight_name', 'weight_device'}
        need(type(value) is dict and set(value) == fields, 'malformed node record')
        need(type(value['name']) is str and 0 < len(value['name']) <= 256, 'invalid node name')
        need(value['op'] == op and value['phase'] == 'post_execution' and value['executed'] is True,
             'node was not observed after execution')
        device(value['output_device'], 'either' if weight == 'token_embd.weight' and backend == 'cuda' else backend)
        if weight is not None:
            need(value['weight_name'] == weight, 'wrong observed weight')
            device(value['weight_device'], 'cpu' if weight == 'token_embd.weight' else backend)

    fa = evidence['flash_attention_nodes']
    need(type(fa) is list and len(fa) == expected_layers, 'incomplete FA layer coverage')
    for value in fa:
        node(value, 'FLASH_ATTN_EXT')
    need(len({value['name'] for value in fa}) == expected_layers, 'duplicate FA node')
    for field, op, weight in [('classifier_nodes', 'MUL_MAT', 'cls.output.weight'),
                              ('token_embedding_nodes', 'GET_ROWS', 'token_embd.weight')]:
        values = evidence[field]
        need(type(values) is list and len(values) == 1, 'missing/duplicate ' + field)
        node(values[0], op, weight)
    need(type(evidence['vocabulary_matmul_nodes']) is list and not evidence['vocabulary_matmul_nodes'],
         'unexpected vocabulary matmul')
    return evidence


def source_bf16_view(values):
    """Diagnostic-only nearest-even BF16 output view; never used for scoring."""
    rounded = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError('Nonfinite native output cannot be rounded')
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        rounded.append(struct.unpack('<f', struct.pack('<I', bits))[0])
    return rounded


class NativeRunner:
    def __init__(self, runner_path, gguf_path, *, context_size=4096, threads=4,
                 gpu_layers=0, timeout_seconds=600, stderr_path=None, expected_full_attention_layers=16):
        verify_evidence_contract()
        if type(gpu_layers) is not int or gpu_layers not in (0, -1):
            raise ValueError('This versioned backend supports only CPU (0) or all-layer CUDA (-1)')
        if type(expected_full_attention_layers) is not int or expected_full_attention_layers <= 0:
            raise ValueError('Expected full-attention layer count must be a positive integer')
        self.backend = 'cpu' if gpu_layers == 0 else 'cuda'
        self.expected_full_attention_layers = expected_full_attention_layers
        self.timeout = float(timeout_seconds)
        if self.timeout <= 0 or not math.isfinite(self.timeout):
            raise ValueError('timeout_seconds must be positive and finite')
        self._lock = threading.Lock()
        self._id = 0
        self._stderr = open(stderr_path, 'a') if stderr_path else None
        command = [str(Path(runner_path).resolve()), '--model', str(Path(gguf_path).resolve()),
                   '--ctx-size', str(context_size), '--threads', str(threads),
                   '--gpu-layers', str(gpu_layers), '--timeout-ms', str(int(self.timeout * 1000))]
        self.command = command
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self._stderr, bufsize=0)
        except BaseException:
            if self._stderr is not None:
                self._stderr.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def evaluate(self, token_ids, readout_token_ids):
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(f'Native GLiDE runner exited ({self.process.returncode})')
            self._id += 1
            request = {'id': self._id, 'token_ids': token_ids, 'readout_token_ids': readout_token_ids}
            payload = (json.dumps(request, separators=(',', ':'), allow_nan=False) + '\n').encode()
            deadline = time.monotonic() + self.timeout
            sent, received = 0, bytearray()
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(self.process.stdin, selectors.EVENT_WRITE)
                    selector.register(self.process.stdout, selectors.EVENT_READ)
                    while b'\n' not in received:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError('Native GLiDE transaction timed out')
                        events = selector.select(remaining)
                        if not events:
                            raise TimeoutError('Native GLiDE transaction timed out')
                        for key, _ in events:
                            if key.fileobj is self.process.stdin:
                                sent += os.write(self.process.stdin.fileno(), payload[sent:sent + 65536])
                                if sent == len(payload):
                                    selector.unregister(self.process.stdin)
                            else:
                                chunk = os.read(self.process.stdout.fileno(), 65536)
                                if not chunk:
                                    raise RuntimeError('Native GLiDE runner closed stdout before a complete result')
                                received.extend(chunk)
                                if len(received) > 1 << 20:
                                    raise RuntimeError('Native GLiDE response exceeds 1 MiB')
                line, remainder = received.split(b'\n', 1)
                if remainder.strip() or sent != len(payload):
                    raise RuntimeError('Unexpected extra or premature native output')
                response = strict_json(line)
            except BaseException:
                # Discard uncertain protocol state; do not reuse a hung or
                # partially written/read transaction for the next question.
                self.close()
                raise
            if not isinstance(response, dict):
                self.close()
                raise RuntimeError('Native runner response is not an object')
            if type(response.get('id')) is not int or response['id'] != self._id:
                self.close()
                raise RuntimeError('Native runner response id mismatch')
            if response.get('ok') is not True:
                self.close()
                raise RuntimeError(f'Native GGUF inference failed: {response.get("error")}')
            row = response.get('logits')
            if (not isinstance(row, list) or len(row) != 255 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in row)):
                self.close()
                raise RuntimeError('Native runner returned invalid readout logits')
            if type(response.get('prompt_tokens')) is not int or response['prompt_tokens'] != len(token_ids):
                self.close()
                raise RuntimeError('Native runner token count changed')
            try:
                validate_execution_evidence(response.get('execution_evidence'), prompt_tokens=len(token_ids),
                                            expected_layers=self.expected_full_attention_layers, backend=self.backend)
            except BaseException:
                self.close()
                raise
            return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout, self._stderr):
            if stream is not None:
                stream.close()


class GlideGGUF:
    """Compatible text-only decide/decide_batch surface; execution is serial.

    Unlike official HF batching, decide_batch invokes one native forward per
    question. Record this execution difference for timing comparisons.
    """
    def __init__(self, source_dir, *, runner=None, tokenizer=None, max_len=4096, name='GLiDE-GGUF-FA-hostemb-v1'):
        self.source_dir = Path(source_dir)
        source_config, self.config = validate_source_files(source_dir)
        self.expected_full_attention_layers = source_config['text_config']['layer_types'].count('full_attention')
        self.prompt = load_prompt_module(source_dir)
        self._softmax, self._answer = load_official_postprocessing(source_dir)
        self.RequestError = self.prompt.RequestError
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer
        self.codes = self.config['codes']
        self.readout_token_ids = self.config['token_ids']
        for code, token_id in zip(self.codes, self.readout_token_ids):
            if tokenizer.encode(code, add_special_tokens=False) != [token_id]:
                raise ValueError(f'Decision code/tokenizer mismatch: {code}')
        self.prompter = self.prompt.Prompter(tokenizer, self.codes)
        self.temperature = float(self.config['temperature'])
        if type(max_len) is not int or not 16 <= max_len <= 40000:
            raise ValueError('max_len must be an integer between 16 and 40000')
        self.max_len = max_len
        self.name = name
        self.vision = False
        self.runner = runner
        self.last_trace = []

    @classmethod
    def from_pretrained(cls, source_dir, *, gguf_path, runner_path, context_size=4096,
                        threads=4, gpu_layers=0, timeout_seconds=600,
                        stderr_path=None, name='GLiDE-GGUF-FA-hostemb-v1'):
        # Validate sources before starting a native process or loading weights.
        instance = cls(source_dir, max_len=context_size, name=name)
        instance.runner = NativeRunner(runner_path, gguf_path, context_size=context_size,
                                       threads=threads, gpu_layers=gpu_layers,
                                       timeout_seconds=timeout_seconds, stderr_path=stderr_path,
                                       expected_full_attention_layers=instance.expected_full_attention_layers)
        return instance

    def prepare_request(self, req):
        if not isinstance(req, dict):
            raise self.RequestError('request must be an object')
        questions = req.get('questions')
        if not isinstance(questions, dict):
            raise self.RequestError('questions must be an object mapping keys to questions')
        state, refs = self.prompt.split_state(req.get('state'))
        refs = list(refs) + list(req.get('images') or [])
        if refs:
            raise self.RequestError('images are not supported by this deployment')
        entries = []
        for key, question in questions.items():
            options = self.prompt.options_for(question)
            if len(options) > 255:
                raise self.RequestError(f'options per choice exceeds the label set ({len(options)} > 255)')
            text = self.prompter.text(state, question, options, None)
            ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
            # Match official reserved-length test, not an undocumented truncation.
            if len(ids) + 8 > self.max_len:
                raise self.RequestError('prompt exceeds the maximum context length')
            if not text.endswith('Answer: (') or not ids:
                raise RuntimeError('Pinned GLiDE answer-cue invariant failed')
            entries.append({'question_id': key, 'type': question['type'],
                            'keys': [k for k, _ in options], 'text': text,
                            'token_ids': ids,
                            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                            'token_ids_sha256': hashlib.sha256(canonical_ids(ids).encode()).hexdigest()})
        return entries

    def decide(self, state, questions, images=None):
        req = {'state': state, 'questions': questions}
        if images:
            req['images'] = images
        response = self.decide_batch([req])[0]
        if isinstance(response, self.RequestError):
            raise response
        return response

    def decide_batch(self, requests):
        if self.runner is None:
            raise RuntimeError('No native GGUF runner was configured')
        results, trace = [], []
        for request in requests:
            try:
                entries = self.prepare_request(request)
            except self.RequestError as exc:
                results.append(exc)
                continue
            answers, usage = {}, 0
            for entry in entries:
                response = self.runner.evaluate(entry['token_ids'], self.readout_token_ids)
                try:
                    evidence = validate_execution_evidence(
                        response.get('execution_evidence'), prompt_tokens=len(entry['token_ids']),
                        expected_layers=self.expected_full_attention_layers,
                        backend=getattr(self.runner, 'backend', 'cuda'))
                except BaseException:
                    self.close()
                    raise
                probabilities = self._softmax(response['logits'], len(entry['keys']), self.temperature)
                answers[entry['question_id']] = self._answer(entry['type'], entry['keys'], probabilities)
                usage += len(entry['token_ids'])
                trace.append({k: entry[k] for k in ['question_id', 'prompt_sha256', 'token_ids_sha256']} |
                             {'logits': response['logits'], 'prompt_tokens': len(entry['token_ids']),
                              'decode_ms': response.get('decode_ms'), 'total_ms': response.get('total_ms')})
                trace[-1]['source_bf16_rounded_logits_diagnostic'] = source_bf16_view(response['logits'])
                trace[-1]['scoring_policy'] = 'native_raw_f32_readout_softmax_no_output_rounding'
                trace[-1]['execution_evidence'] = evidence
                trace[-1]['backend_profile'] = BACKEND_PROFILE
            results.append({'model': self.name, 'answers': answers,
                            'usage': {'input_tokens': usage, 'output_tokens': 0}})
        self.last_trace = trace
        return results

    def close(self):
        if self.runner is not None:
            self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
