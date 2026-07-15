#!/usr/bin/env bash
set -euo pipefail

# Generate frozen Prompt 6 schema-condition configs from the already frozen
# Prompt 3B/P4 protocol. The source configuration contains every economic,
# learning, and evolution setting; this script changes only the observation
# schema version and opt-in rolling provider fields.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="$root/configs/prompt6"
mkdir -p "$output"

copy_base() {
    local source="$1"
    local target="$2"
    sed 's/^version = 2$/version = 3/' "$source" > "$target"
}

append_rolling() {
    local target="$1"
    local returns="$2"
    local volatility="$3"
    local sma="$4"
    local relative="$5"
    local summary="$6"
    cat >> "$target" <<EOF

# Prompt 6 opt-in market senses. Raw values are normalised by the schema.
[observations]
schema = "default"
providers = ["market", "portfolio", "agent_state", "rolling_market_stats"]

[observations.rolling_market_stats]
enabled = true
return_windows = $returns
volatility_windows = $volatility
sma_windows = $sma
relative_strength_windows = $relative
market_summary = $summary
market_summary_window = 20
minimum_history = "zero_fill"
EOF
}

# A is exactly the default registry-backed Prompt 5 schema. B--D add only the
# listed sensor groups. F intentionally uses the existing evolution condition.
cp "$root/configs/prompt3b/runs/control-seed-41.toml" "$output/default-control.toml"
cp "$root/configs/prompt3b/runs/ppo-seed-41.toml" "$output/default-ppo.toml"

copy_base "$root/configs/prompt3b/runs/ppo-seed-41.toml" "$output/momentum-ppo.toml"
append_rolling "$output/momentum-ppo.toml" '[20, 60]' '[]' '[]' '[20, 60]' 'false'

copy_base "$root/configs/prompt3b/runs/ppo-seed-41.toml" "$output/volatility-trend-ppo.toml"
append_rolling "$output/volatility-trend-ppo.toml" '[]' '[20, 60]' '[60]' '[]' 'true'

copy_base "$root/configs/prompt3b/runs/ppo-seed-41.toml" "$output/combined-ppo.toml"
append_rolling "$output/combined-ppo.toml" '[5, 20, 60, 120]' '[20, 60]' '[20, 60]' '[20, 60]' 'true'

copy_base "$root/configs/prompt3b/runs/ppo-evolution-seed-41.toml" "$output/combined-ppo-evolution.toml"
append_rolling "$output/combined-ppo-evolution.toml" '[5, 20, 60, 120]' '[20, 60]' '[20, 60]' '[20, 60]' 'true'
