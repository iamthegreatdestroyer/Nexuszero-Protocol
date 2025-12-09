#!/usr/bin/env python3
import re
import json
from pathlib import Path

# Files to parse
logs = {
    'crypto': 'nexuszero-crypto/benches/crypto_benchmarks_output.log',
    'bulletproof': 'nexuszero-crypto/benches/bulletproof_benchmarks_output.log',
    'proof': 'nexuszero-crypto/benches/proof_benchmarks_output.log'
}

baseline_path = Path('benchmark_summary.json')
if not baseline_path.exists():
    print('No baseline found at benchmark_summary.json')
    exit(1)

baseline = json.loads(baseline_path.read_text())
bench_baseline = {b['name']: b for b in baseline['benchmarks']}

results = {}

# Regex to find benchmark time line: time:   [min unit mean unit max unit]
# Match the inner part inside the brackets to be more tolerant of unicode and spacing: 'time: [ inner ]'
pat = re.compile(r"\s*time:\s*\[(?P<inner>[^\]]+)\]")
# Map micro symbol alt 'μs' replace
unit_map = {'ns':1/1e3, 'μs':1, 'us':1, 'ms':1000}

for key, path in logs.items():
    p = Path(path)
    if not p.exists():
        continue
    text = p.read_text()
    # Find all benchmark names from lines that read 'Benchmarking <name>'
    names = re.findall(r'Benchmarking\s+([a-zA-Z0-9_\-]+)', text)
    # Then find all 'time:' lines (inner text inside brackets)
    times = pat.findall(text)
    print(f"[DEBUG] Parsing log: {path}, found {len(names)} benchmark names and {len(times)} time entries")
    # pair names and times in found order
    for idx, name in enumerate(names):
        if idx < len(times):
            inner = times[idx]
            parts = re.split(r"\\s+", inner.strip())
            # parts are like ['422.97', '┬╡s', '444.30', '┬╡s', '467.47', '┬╡s'] or ['1.5397', 'ms'] etc.
            nums = []
            units = []
            for p in parts:
                m = re.match(r"([0-9]*\\.?[0-9]+)(.*)", p)
                if m:
                    nums.append(float(m.group(1)))
                    unit = re.sub(r"[^a-zA-Zμ]", "", m.group(2))
                    units.append(unit)
            # Determine mean value (prefer central element)
            if len(nums) >= 3:
                mean_val = nums[1]
                mean_unit = units[1] if units[1] != '' else (units[0] if units[0] != '' else 'us')
            elif len(nums) == 1:
                mean_val = nums[0]
                mean_unit = units[0] if units[0] != '' else 'us'
            else:
                continue
            mean = float(mean_val) * unit_map.get(mean_unit,1)
            # Try to fill reasonable min/max if not present
            min_v = nums[0] if len(nums) >= 1 else mean_val
            max_v = nums[2] if len(nums) >= 3 else mean_val
            results[name] = {
                'mean_us': mean,
                'min': float(min_v),
                'max': float(max_v),
                'unit': mean_unit
            }

# Compute relative change vs baseline
report = []
for name, data in results.items():
    if name in bench_baseline:
        base = bench_baseline[name]
        baseline_us = base['mean_us']
        change = (data['mean_us'] - baseline_us)/baseline_us * 100.0
        report.append({
            'benchmark': name,
            'baseline_mean_us': baseline_us,
            'current_mean_us': data['mean_us'],
            'relative_change_pct': change
        })
    else:
        report.append({
            'benchmark': name,
            'baseline_mean_us': None,
            'current_mean_us': data['mean_us'],
            'relative_change_pct': None
        })

# Write report
out = Path('docs/benchmark_summary_current.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'report': report}, indent=2))
print('Wrote docs/benchmark_summary_current.json')
print('Done')
