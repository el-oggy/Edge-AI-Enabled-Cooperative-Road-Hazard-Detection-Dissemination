"""Reproducible M1 verification; saves unique CLI logs, raw outputs and commands.
Run from project root with .venv\\Scripts\\python.exe -B this_file.py.
Synthetic corrupt resources and mock fixtures are explicitly identified below.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
RUNNER = ROOT / 'src' / 'edge_rpi_hazard_inference.py'
records = []


def run(label, arguments, expected, cwd=ROOT):
    command = [str(PYTHON), '-B', *arguments]
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    stdout = HERE / (label + '.stdout.json')
    stderr = HERE / (label + '.stderr.log')
    stdout.write_text(completed.stdout, encoding='utf-8')
    stderr.write_text(completed.stderr, encoding='utf-8')
    entry = {'label': label, 'command_argv': command,
             'powershell_command': '& ' + ' '.join("'" + part.replace("'", "''") + "'" for part in command),
             'cwd': str(cwd), 'exit_code': completed.returncode, 'expected_exit_code': expected,
             'expectation_satisfied': completed.returncode == expected,
             'wall_seconds': time.perf_counter() - start,
             'stdout': str(stdout), 'stderr': str(stderr)}
    records.append(entry)
    print(label, completed.returncode, 'expected', expected, flush=True)
    (HERE / 'verification_commands.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return completed


output_arguments = ['--conf', '0.35', '--iou', '0.45', '--output-dir', str(HERE)]
for domain, image in [('pothole', 'test_user_pothole.png'), ('construction', 'test_user_construction.png'),
                       ('accident_obstacle', 'test_user_accident.png')]:
    run(domain + '_matched_scene', [str(RUNNER), '--model', 'quantization/' + domain + '_yolo11s_int8.onnx',
                                  '--image', 'data/' + image, *output_arguments], 0)
    run(domain + '_unannotated_road', [str(RUNNER), '--model', 'quantization/' + domain + '_yolo11s_int8.onnx',
                                  '--image', 'data/sample_pothole_verified.jpg', *output_arguments], 0)

run('real_image_benchmark_smoke', [str(RUNNER), '--image', 'data/test_user_pothole.png',
                                  '--benchmark', '--iterations', '1', '--warmup', '0', *output_arguments], 0)
run('zero_tensor_benchmark_smoke', [str(RUNNER), '--benchmark', '--iterations', '1', '--warmup', '0'], 0)
run('default_model_outside_cwd', [str(RUNNER), '--image', 'data/test_user_pothole.png', *output_arguments], 0, HERE)
run('module_help', ['-m', 'src.edge_rpi_hazard_inference', '--help'], 0)
run('missing_model', [str(RUNNER), '--model', 'quantization/M1_ABSENT.onnx', '--image', 'data/test_user_pothole.png'], 1)
run('missing_image', [str(RUNNER), '--image', 'data/M1_ABSENT.png'], 1)
corrupt = HERE / 'synthetic_corrupt_input.png'
corrupt.write_bytes(b'M1 synthetic invalid image, not road footage')
run('corrupt_image', [str(RUNNER), '--image', str(corrupt)], 1)
unreviewed = HERE / 'synthetic_unreviewed_model.onnx'
unreviewed.write_bytes(b'M1 synthetic invalid model, not trained weights')
run('unreviewed_model', [str(RUNNER), '--model', str(unreviewed), '--image', 'data/test_user_pothole.png'], 1)
run('mock_stream', [str(RUNNER), '--simulate-stream'], 0)
run('mixed_mock_real', [str(RUNNER), '--simulate-stream', '--image', 'data/test_user_pothole.png'], 2)
run('invalid_threshold', [str(RUNNER), '--image', 'data/test_user_pothole.png', '--conf', 'nan'], 2)
run('no_action', [str(RUNNER)], 2)
run('unsupported_benchmark_output', [str(RUNNER), '--benchmark', '--output-dir', str(HERE)], 2)

summaries = []
for entry in records:
    if entry['exit_code'] != 0 or 'matched_scene' not in entry['label'] and 'unannotated_road' not in entry['label']:
        continue
    result = json.loads(Path(entry['stdout']).read_text(encoding='utf-8'))
    summaries.append({'label': entry['label'], 'model': result['model']['source_model_id'],
                      'input': result['input']['path'], 'raw_prediction': result['raw_prediction'],
                      'timings_ms': result['timings_ms'], 'detections': result['detections'],
                      'artifacts': result['artifacts']})
(HERE / 'smoke_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
if not all(entry['expectation_satisfied'] for entry in records):
    raise SystemExit(1)
