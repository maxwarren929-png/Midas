#!/usr/bin/env python3
"""Monitor the 14 Camber jobs, sync artifacts, and print the complete comparison leaderboard."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix"
REPORTS_DIR = ROOT / "reports" / "camber_matrix"

JOB_MAP = {
    23896: ("01_baseline_ppo", 41),
    23897: ("01_baseline_ppo", 42),
    23898: ("02_adamw_optimizer", 41),
    23899: ("02_adamw_optimizer", 42),
    23900: ("03_deep_recurrent", 41),
    23901: ("03_deep_recurrent", 42),
    23902: ("04_drawdown_penalty", 41),
    23903: ("04_drawdown_penalty", 42),
    23904: ("05_turnover_penalty", 41),
    23905: ("05_turnover_penalty", 42),
    23906: ("06_ppo_evolution", 41),
    23907: ("06_ppo_evolution", 42),
    23908: ("07_winning_combo", 41),
    23910: ("07_winning_combo", 42),
}

def run_cmd(cmd):
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return res.stdout.strip()

print(f"Monitoring {len(JOB_MAP)} jobs on Camber Cloud...")
pending = list(JOB_MAP.keys())

while pending:
    still = []
    for jid in pending:
        out = run_cmd(["camber", "job", "get", str(jid)])
        m = re.search(r"Status:\s*([A-Z_]+)", out)
        st = m.group(1) if m else "UNKNOWN"
        cond, seed = JOB_MAP[jid]
        if st in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"  [+] Job {jid} ({cond}, seed {seed}) finished with status: {st}")
        else:
            still.append(jid)
    pending = still
    if pending:
        print(f"  --> {len(pending)} / {len(JOB_MAP)} jobs still training in the cloud...")
        time.sleep(12)

print("\nAll jobs finished! Syncing artifacts from Stash...")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["PATH"] = f"{Path.home()}/.camber/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
subprocess.run(["camber", "stash", "cp", "-r", "stash://kohosting/projects/midas/checkpoints/camber_matrix/", str(CHECKPOINTS_DIR)], env=env)
subprocess.run(["camber", "stash", "cp", "-r", "stash://kohosting/projects/midas/reports/camber_matrix/", str(REPORTS_DIR)], env=env)

# Run Evaluation and format leaderboard
results = []
BIN = ROOT / "target" / "release" / "midas-cli"
CONFIGS_DIR = ROOT / "configs" / "camber_matrix"

for jid, (cond, seed) in JOB_MAP.items():
    cp = CHECKPOINTS_DIR / f"{cond}-seed-{seed}.bin"
    cfg = CONFIGS_DIR / f"{cond}-seed-{seed}.toml"
    if not cp.exists() or not cfg.exists():
        continue
    proc = subprocess.run([str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"], capture_output=True, text=True)
    if proc.returncode != 0:
        continue
    try:
        data = json.loads(proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            sharpes = [a[1]["sharpe_like"] for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            results.append({
                "condition": cond,
                "seed": seed,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "sharpe_like": sum(sharpes) / len(sharpes),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
            })
    except Exception as e:
        print(f"Error parsing {cond}-{seed}: {e}")

# Group by condition
conds = {}
for r in results:
    conds.setdefault(r["condition"], []).append(r)

print("\n" + "="*95)
print(f"{'Condition':<26} | {'Seeds':<5} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
print("="*95)
for cond, rows in sorted(conds.items()):
    n = len(rows)
    avg_ret = sum(r["return_%"] for r in rows) / n
    avg_dd = sum(r["max_drawdown_%"] for r in rows) / n
    avg_to = sum(r["turnover_$"] for r in rows) / n
    avg_fric = sum(r["friction_$"] for r in rows) / n
    print(f"{cond:<26} | {n:<5} | {avg_ret:>+15.2f}% | {avg_dd:>10.2f}% | ${avg_to:>10.2f} | ${avg_fric:>10.2f}")
print("="*95)
