"""Freeze independent synthetic protocol controls before any GPU results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from glide_gguf import GlideGGUF
from glide_contract import sha256_file, write_json


def cases():
    result = []
    def add(name, state, question):
        result.append({'id': name, 'request': {'state': state, 'questions': {'q': question}},
                       'source': 'new synthetic compatibility control; not a public benchmark'})
    for i, color in enumerate(['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple']):
        add(f'choice8_color_{i}', f'The label explicitly says {color}.',
            {'type': 'choice', 'instructions': 'Choose the stated color.',
             'criteria': {c: c for c in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple']}})
    for i, text in enumerate(['The statement is true.', 'The statement is false.', '', '是，答案为真。',
                              'No, this is incorrect.', 'Yes, it is correct.', '不确定。', 'A = A.']):
        add(f'noul_{i}', text, {'type': 'noul', 'instructions': 'Does the state affirm the statement?',
                               'criteria': {'true': 'Affirmed / 是', 'false': 'Not affirmed / 否'}})
    for i, text in enumerate(['Terrible.', 'Neutral.', 'Excellent.', '一般。']):
        add(f'score3_{i}', text, {'type': 'score', 'instructions': 'Rate the sentiment.',
                                'criteria': ['negative', 'neutral', 'positive']})
    add('choice2_unicode', {'城市': '北京', 'languages': ['中文', 'English'], 'emoji': '😀'},
        {'type': 'choice', 'instructions': {'question': 'Which city is stated?', 'rule': ['use state']},
         'criteria': {'北京': '北京', '上海': '上海'}})
    add('choice3_json_options', ['x', None, True],
        {'type': 'choice', 'instructions': ['Pick the closest representation.', 1],
         'criteria': {'first': {'x': 1}, 'second': None, 'third': False}})
    add('score_numeric_keys', {'rating': 2}, {'type': 'score', 'instructions': 'Rate the value.',
                                            'criteria': {'10': 'high', '2': 'middle', '1': 'low'}})
    add('null_state', None, {'type': 'choice', 'instructions': '', 'criteria': {'one': 'one', 'two': 'two'}})
    add('all255_options', 'The exact target is key137.',
        {'type': 'choice', 'instructions': 'Select the exact target.',
         'criteria': {f'key{i}': f'key{i}' for i in range(255)}})
    for repeats in [500, 1200, 2000]:
        add(f'long_context_{repeats}', 'Start: target is amber.\n' + 'neutral filler. ' * repeats + '\nEnd: target remains amber.',
            {'type': 'choice', 'instructions': 'What is the target?', 'criteria': {'amber': 'amber', 'violet': 'violet'}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    m = GlideGGUF(a.source_dir, max_len=40000)
    frozen = []
    for case in cases():
        entry = m.prepare_request(case['request'])[0]
        frozen.append(case | {'prompt': entry['text'], 'token_ids': entry['token_ids'],
                              'prompt_sha256': entry['prompt_sha256'], 'token_ids_sha256': entry['token_ids_sha256'],
                              'prompt_tokens': len(entry['token_ids']), 'option_count': len(entry['keys']),
                              'question_type': entry['type'], 'option_keys': entry['keys']})
    if len(frozen) < 24 or max(x['prompt_tokens'] for x in frozen) < 5261:
        raise RuntimeError('Insufficient frozen diagnostic coverage')
    if max(x['prompt_tokens'] for x in frozen) + 8 > 8192:
        raise RuntimeError('Frozen synthetic case exceeds planned 8192 context')
    a.output_dir.mkdir(parents=True)
    file = a.output_dir / 'cases.jsonl'
    with file.open('x') as f:
        for case in frozen:
            f.write(json.dumps(case, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    write_json(a.output_dir / 'manifest.json', {
        'schema': 'glide.synthetic_parity_cases.v1', 'status': 'frozen_before_gpu_execution',
        'cases_file': file.name, 'cases_sha256': sha256_file(file), 'case_count': len(frozen),
        'question_types': sorted(set(x['question_type'] for x in frozen)),
        'option_counts': sorted(set(x['option_count'] for x in frozen)),
        'prompt_tokens': {x['id']: x['prompt_tokens'] for x in frozen},
        'reset_sequence': ['choice8_color_0', 'choice8_color_1', 'choice8_color_0'],
        'scoring_policy': 'native_raw_f32_readout_softmax_no_output_rounding',
        'not_for_quality_tuning': True, 'not_a_public_benchmark': True,
        'generator_sha256': sha256_file(__file__), 'source_revision': m.config.get('revision', '1a9a7ed94379c2ea6f324176c9bc22fb9b7b7d60')})
    print(json.dumps({'cases': len(frozen), 'max_prompt_tokens': max(x['prompt_tokens'] for x in frozen),
                      'sha256': sha256_file(file)}))


if __name__ == '__main__':
    main()
