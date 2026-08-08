# V3 Frozen-Encoder Baseline

Captured before V3 autodiff or encoder-training changes.

## Reproduction contract

- Source commit: `c03847548c7ede76228786c4e83e912bb9ed2a07`
- Configuration: `configs/prompt6/combined-ppo.toml`
- Configuration identity: `d72abfff021f7f37c1659e626a82c9f05fe937f5418e31994e28cccf55c9e1bd`
- Dataset identity: `638b9286f2c8965e68f8212d97cda6102a99d9337dd83e76b6ed52c6794464aa`
- Observation schema identity: `ceca739b4fac89b487a2a872e5d973632fb6e4b4e605e84d34cf8e17d64a316e`
- Split: training
- Seed: 41
- Population: 8
- Architecture: hidden `[32, 16]`, simple recurrent memory 8
- Learning: frozen-encoder PPO actor/critic head updates
- Observation provider set: market, portfolio, agent state, rolling market statistics
- Boundary: 127 completed steps. This deliberately stops before the configured
  128-step automatic checkpoint so the pre-existing Prompt 3B checkpoint is not
  overwritten.
- Build profile: Cargo `dev` profile with debug information
- Rust: `rustc 1.96.1 (31fca3adb 2026-06-26)`
- CPU: AMD Ryzen 5 7520U, 4 cores / 8 threads

Command:

```bash
/tmp/midas-baseline-target/debug/midas-cli simulate \
  --config configs/prompt6/combined-ppo.toml \
  --split training \
  --steps 127 \
  --checkpoint checkpoints/baselines/frozen-combined-ppo-seed-41-step127.bin
```

## Aggregate result

| Metric | Value |
|---|---:|
| Completed steps | 127 |
| Agents | 8 |
| Mean ending net worth | 10,130.2926 |
| Mean total return | 1.30293% |
| PPO learning updates | 248 |
| Neural actions | 1,016 |
| Total trades | 3,808 |
| Deaths | 0 |
| Invalid actions | 0 |
| Multi-asset action fraction | 98.9173% |
| Mean active positions | 3.60039 |
| Mean concentration | 0.272621 |
| Mean cash allocation | 0.452874 |
| Mean absolute trade impulse | 0.537277 |

Per-agent ending net worth:

| Agent | Ending net worth | Total return |
|---:|---:|---:|
| 0 | 10,174.4186 | 1.74419% |
| 1 | 10,149.6673 | 1.49667% |
| 2 | 10,097.2075 | 0.97207% |
| 3 | 10,212.2322 | 2.12232% |
| 4 | 10,175.0248 | 1.75025% |
| 5 | 9,941.6485 | -0.58351% |
| 6 | 10,046.5552 | 0.46555% |
| 7 | 10,245.5865 | 2.45587% |

## Resource baseline

The first two timed runs completed in 5.943 and 5.945 seconds. The first run
used 13.091 user CPU seconds and 2.188 system CPU seconds. A third replay with
20 ms `/proc` sampling completed in 6.733 seconds and reached a sampled peak
resident set size of 44,632 KiB (43.59 MiB). Compilation time is excluded.

These numbers describe the debug build and should only be compared with V3
runs built using the same profile and environment. Release-mode performance
must be captured separately if it becomes the experiment profile.

## Determinism and checkpoint

Three independent runs produced byte-identical checkpoints:

- Checkpoint: `checkpoints/baselines/frozen-combined-ppo-seed-41-step127.bin`
- Size: 879,409 bytes
- SHA-256: `e31c8430857e51c87582e55d573087f5f7976ccdb34e0aec57959f4c5d5cb597`

The complete JSON simulation report was also identical across the first two
runs. This checkpoint and the identities above form the regression boundary
for the first V3 autodiff-equivalence stage.
