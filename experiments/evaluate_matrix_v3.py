#!/usr/bin/env python3
"""Evaluate all 12 Matrix V3 checkpoints on the held-out test split."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "midas-cli"
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v3"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v3"

summary = []

for cp in sorted(CHECKPOINTS_DIR.glob("*.bin")):
    name = cp.stem
    cfg = CONFIGS_DIR / f"{name}.toml"
    if not cfg.exists():
        continue
        
    proc = subprocess.run([str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error evaluating {name}: {proc.stderr}")
        continue
        
    try:
        data = json.loads(proc.stdout)
        agents = data.get("agents", [])
        if agents:
            rets = [a[1]["total_return"] * 100 for a in agents]
            dds = [a[1]["maximum_drawdown"] * 100 for a in agents]
            sharpes = [a[1]["sharpe_like"] for a in agents]
            tos = [a[1]["turnover"] for a in agents]
            trades = [a[1]["trades"] for a in agents]
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            bankruptcies = sum(1 for a in agents if a[1]["bankrupt"])
            
            cond_name = "-".join(name.split("-")[:-2])
            seed = int(name.split("-")[-1])
            
            summary.append({
                "condition": cond_name,
                "seed": seed,
                "return_%": sum(rets) / len(rets),
                "max_drawdown_%": sum(dds) / len(dds),
                "sharpe_like": sum(sharpes) / len(sharpes),
                "turnover_$": sum(tos) / len(tos),
                "friction_$": sum(fees) / len(fees),
                "trades": sum(trades) / len(trades),
                "bankruptcies": bankruptcies
            })
    except Exception as e:
        print(f"Error parsing {name}: {e}")

# Group by condition
cond_groups = {}
for row in summary:
    cond_groups.setdefault(row["condition"], []).append(row)

leaderboard = []
for cond, rows in cond_groups.items():
    n = len(rows)
    avg_ret = sum(r["return_%"] for r in rows) / n
    avg_dd = sum(r["max_drawdown_%"] for r in rows) / n
    avg_to = sum(r["turnover_$"] for r in rows) / n
    avg_fric = sum(r["friction_$"] for r in rows) / n
    leaderboard.append((avg_ret, cond, n, avg_dd, avg_to, avg_fric))

leaderboard.sort(key=lambda x: x[0], reverse=True)

print("\n" + "="*115)
print(f"{'Condition':<40} | {'Seeds':<5} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
print("="*115)
for avg_ret, cond, n, avg_dd, avg_to, avg_fric in leaderboard:
    print(f"{cond:<40} | {n:<5} | {avg_ret:>+15.2f}% | {avg_dd:>10.2f}% | ${avg_to:>10.2f} | ${avg_fric:>10.2f}")
print("="*115)
