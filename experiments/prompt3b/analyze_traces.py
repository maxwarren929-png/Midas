#!/usr/bin/env python3
"""Convert committed Prompt 3B JSONL traces into compact CSV diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--assets", default="SPY,AAA,BBB,CCC")
    args = parser.parse_args()
    assets = args.assets.split(",")
    decisions: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    with (args.trace_dir / "traces.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if "Decision" in value:
                decisions.append(value["Decision"])
            elif "PpoUpdate" in value:
                updates.append(value["PpoUpdate"])

    action_rows = []
    reward_rows = []
    cost_rows = []
    order_rows = []
    for row in decisions:
        action_rows.append(
            {
                "observation_date": row["observation_date"],
                "execution_date": row["execution_date"],
                "agent_id": row["agent_id"],
                "genome_id": row["genome_id"],
                "generation_depth": row["generation_depth"],
                "policy": json.dumps(row["policy"], separators=(",", ":")),
                "previous_net_worth": row["previous_net_worth"],
                "current_net_worth": row["current_net_worth"],
                "reward": row["reward"],
                "cash_allocation": row["cash_allocation"],
                "maximum_concentration": row["maximum_concentration"],
                "active_positions": row["active_positions"],
                "turnover": row["turnover"],
                "trade_count": row["trade_count"],
                "buy_notional": row["buy_notional"],
                "sell_notional": row["sell_notional"],
                "entropy": row["entropy"],
                "old_log_probability": row["old_log_probability"],
                "value_estimate": row["value_estimate"],
                "temperature": row["temperature"],
                "invalid_action": row["invalid_action"],
                "failure_category": row["failure_category"],
                "actor_eligible": row["actor_eligible"],
                "critic_eligible": row["critic_eligible"],
                "current_allocations": json.dumps(row["current_allocations"], separators=(",", ":")),
                "action_allocations": json.dumps(row["action_allocations"], separators=(",", ":")),
                "active_asset_indices": json.dumps(row["active_asset_indices"], separators=(",", ":")),
                "latent_samples": json.dumps(row["latent_samples"], separators=(",", ":")),
            }
        )
        breakdown = row["reward_breakdown"]
        reward_rows.append(
            {
                "observation_date": row["observation_date"],
                "execution_date": row["execution_date"],
                "agent_id": row["agent_id"],
                **breakdown,
                "reward": row["reward"],
            }
        )
        cost_rows.append(
            {
                "observation_date": row["observation_date"],
                "execution_date": row["execution_date"],
                "agent_id": row["agent_id"],
                "turnover": row["turnover"],
                "trade_count": row["trade_count"],
                "buy_notional": row["buy_notional"],
                "sell_notional": row["sell_notional"],
                "transaction_cost": row["transaction_cost"],
                "commission_cost": row["commission_cost"],
                "spread_cost": row["spread_cost"],
                "slippage_cost": row["slippage_cost"],
            }
        )
        for position, (asset_index, allocation) in enumerate(
            zip(row["active_asset_indices"], row["order_position_allocations"])
        ):
            order_rows.append(
                {
                    "observation_date": row["observation_date"],
                    "execution_date": row["execution_date"],
                    "agent_id": row["agent_id"],
                    "order_position": position,
                    "asset_index": asset_index,
                    "asset": assets[int(asset_index)],
                    "allocation": allocation,
                }
            )

    write_csv(args.trace_dir / "action-traces.csv", action_rows)
    write_csv(args.trace_dir / "reward-components.csv", reward_rows)
    write_csv(args.trace_dir / "trading-costs.csv", cost_rows)
    write_csv(args.trace_dir / "allocation-by-order.csv", order_rows)
    write_csv(args.trace_dir / "ppo-update-diagnostics.csv", updates)


if __name__ == "__main__":
    main()
