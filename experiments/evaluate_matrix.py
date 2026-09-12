#!/usr/bin/env python3
"""Evaluate all downloaded Camber Matrix checkpoints on the test split and print leaderboard."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "midas-cli"
CONFIGS_DIR = ROOT / "configs" / "camber_matrix"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix"

results = []

for cp in sorted(CHECKPOINTS_DIR.glob("*.bin")):
    name = cp.stem
    cfg = CONFIGS_DIR / f"{name}.toml"
    if not cfg.exists():
        continue
        
    cmd = [str(BIN), "evaluate", "--config", str(cfg), "--checkpoint", str(cp), "--split", "test"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error evaluating {name}: {proc.stderr}")
        continue
        
    try:
        data = json.loads(proc.stdout)
        agents = data.get("agents", [])
        if agents:
            returns = [a[1]["total_return"] * 100 for a in agents]
            drawdowns = [a[1]["maximum_drawdown"] * 100 for a in agents]
            sharpes = [a[1]["sharpe_like"] for a in agents]
            turnovers = [a[1]["turnover"] for a in agents]
            trades = [a[1]["trades"] for a in agents]
            bankruptcies = sum(1 for a in agents if a[1]["bankrupt"])
            fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
            
            results.append({
                "condition": "-".join(name.split("-")[:-2]),
                "seed": int(name.split("-")[-1]),
                "return_%": sum(returns) / len(returns),
                "max_drawdown_%": sum(drawdowns) / len(drawdowns),
                "sharpe_like": sum(sharpes) / len(sharpes),
                "turnover_$": sum(turnovers) / len(turnovers),
                "friction_costs_$": sum(fees) / len(fees),
                "trades": sum(trades) / len(trades),
                "bankruptcies": bankruptcies
            })
    except Exception as e:
        print(f"Failed to parse output for {name}: {e}")

# Group by condition
conds = {}
for r in results:
    conds.setdefault(r["condition"], []).append(r)

print("\n" + "="*85)
print(f"{'Condition':<25} | {'Seeds':<5} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
print("="*85)
for cond, rows in sorted(conds.items()):
    n = len(rows)
    avg_ret = sum(r["return_%"] for r in rows) / n
    avg_dd = sum(r["max_drawdown_%"] for r in rows) / n
    avg_to = sum(r["turnover_$"] for r in rows) / n
    avg_fric = sum(r["friction_costs_$"] for r in rows) / n
    print(f"{cond:<25} | {n:<5} | {avg_ret:>+15.2f}% | {avg_dd:>10.2f}% | ${avg_to:>10.2f} | ${avg_fric:>10.2f}")
print("="*85)
