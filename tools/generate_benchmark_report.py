#!/usr/bin/env python3
import json
from pathlib import Path
import datetime

baseline_path = Path('benchmark_summary.json')
if not baseline_path.exists():
    print('No baseline file available')
    exit(1)

baseline = json.loads(baseline_path.read_text())
bench_baseline = {b['name']: b for b in baseline['benchmarks']}

report_rows = []

for name, base in bench_baseline.items():
    # Look for criterion folder (name sanitized)
    folder = Path('target/criterion') / name
    if folder.exists():
        est_file = folder / 'new' / 'estimates.json'
        if est_file.exists():
            est = json.loads(est_file.read_text())
            mean_ns = est['mean']['point_estimate']
            mean_us = mean_ns / 1000.0
            baseline_mean_us = base['mean_us']
            change_pct = (mean_us - baseline_mean_us) / baseline_mean_us * 100.0
            report_rows.append({
                'name': name,
                'baseline_mean_us': baseline_mean_us,
                'current_mean_us': mean_us,
                'relative_change_pct': change_pct
            })
        else:
            report_rows.append({'name':name, 'baseline_mean_us': base['mean_us'], 'current_mean_us': None, 'relative_change_pct': None})
    else:
        report_rows.append({'name':name, 'baseline_mean_us': base['mean_us'], 'current_mean_us': None, 'relative_change_pct': None})

# Generate Markdown report
report_md = []
report_md.append('# Benchmark Report')
report_md.append(f'Generated: {datetime.datetime.utcnow().isoformat()} UTC')
report_md.append('\n## Environment')
import platform, subprocess, os
report_md.append(f'- OS: {platform.system()} {platform.release()} {platform.machine()}')
try:
    rustc = subprocess.check_output(['rustc', '--version']).decode().strip()
    cargo = subprocess.check_output(['cargo', '--version']).decode().strip()
    report_md.append(f'- rustc: {rustc}')
    report_md.append(f'- cargo: {cargo}')
except Exception:
    report_md.append('- rustc/cargo: Unknown')

report_md.append('\n## Benchmarks (compared to baseline)')
report_md.append('| Benchmark | Baseline (us) | Current (us) | Change (%) |')
report_md.append('|---|---:|---:|---:|')
for r in report_rows:
    change = f"{r['relative_change_pct']:.2f}%" if r['relative_change_pct'] is not None else 'N/A'
    cur = f"{r['current_mean_us']:.3f}" if r['current_mean_us'] is not None else 'N/A'
    report_md.append(f"| {r['name']} | {r['baseline_mean_us']:.3f} | {cur} | {change} |")

outmd = Path('docs/benchmark_report.md')
outmd.parent.mkdir(parents=True, exist_ok=True)
outmd.write_text('\n'.join(report_md))
print(f'Wrote {outmd}')
outjson = Path('docs/benchmark_summary_current.json')
outjson.write_text(json.dumps({'report': report_rows}, indent=2))
print(f'Wrote {outjson}')
