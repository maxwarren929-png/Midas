#!/usr/bin/env python3
"""
Evaluate all Matrix V5 checkpoints against their respective historical crash events.
"""

import json
import subprocess
from pathlib import Path
from experiments.generate_matrix_v5 import EVENTS, MODELS

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "midas-cli"
CONFIGS_DIR = ROOT / "configs" / "camber_matrix_v5"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "camber_matrix_v5"

for ev_key, ev_info in EVENTS.items():
    print("\n" + "="*115)
    print(f"  HISTORICAL CRASH EVENT: {ev_info['description'].upper()}")
    print("="*115)
    print(f"{'Model Configuration':<36} | {'Test Return (%)':<16} | {'Max DD (%)':<11} | {'Turnover ($)':<12} | {'Friction ($)':<12}")
    print("-"*115)
    
    event_results = []
    for mod_key, mod_info in MODELS.items():
        cp = CHECKPOINTS_DIR / f"{ev_key}__{mod_key}-seed-41.bin"
        cfg = CONFIGS_DIR / f"{ev_key}__{mod_key}-seed-41.toml"
        
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
                tos = [a[1]["turnover"] for a in agents]
                fees = [a[1]["total_fees"] + a[1]["spread_cost"] + a[1]["slippage_cost"] for a in agents]
                
                event_results.append({
                    "model": mod_key,
                    "return_%": sum(rets) / len(rets),
                    "max_drawdown_%": sum(dds) / len(dds),
                    "turnover_$": sum(tos) / len(tos),
                    "friction_$": sum(fees) / len(fees),
                })
        except Exception as e:
            pass

    event_results.sort(key=lambda x: x["return_%"], reverse=True)
    for r in event_results:
        print(f"{r['model']:<36} | {r['return_%']:>+15.2f}% | {r['max_drawdown_%']:>10.2f}% | ${r['turnover_$']:>10.2f} | ${r['friction_$']:>10.2f}")
    if not event_results:
        print("  (Models still downloading / training...)")
    print("="*115)
