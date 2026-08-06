# Midas Experimental Reports & Quantitative Data Synthesis

This document synthesizes the quantitative results, benchmark performance tables, ablation studies, and diagnostic metrics collected across all rigorous experimental validation suites conducted within the Midas test harness.

---

## 1. First Controlled Experiment Suite (Baseline Verification)
* **Dataset & Protocol**: 24-session synthetic market, chronological training/validation/test splits, 5 paired seeds (41–45). Four experimental conditions testing learning and evolutionary activation.

### Mechanical Performance Summary (Training Range Mean of Seed Means)
| Condition | Seeds | Mean Final NW ($) | Median Final NW ($) | Survival Rate | Mean Max Drawdown | PPO Updates | Mean Deaths | Mean Births | Max Gen Depth |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 5 | **10,072.42** | 9,968.23 | 79.5% | 0.00269 | 0.0 | 4.2 | 0.0 | 0 |
| **PPO Only** | 5 | **9,992.60** | 9,991.82 | 73.5% | 0.00447 | 67.8 | 5.8 | 0.0 | 0 |
| **Evolution Only** | 5 | **10,082.67** | 9,968.45 | 83.4% | 0.00262 | 0.0 | 3.2 | 3.2 | 1 |
| **PPO + Evolution** | 5 | **9,992.21** | 9,991.82 | 73.5% | 0.00447 | 67.8 | 5.8 | 5.8 | 2 |

### PPO Head Learning Diagnostic Evidence
| Condition | Actor Delta | Critic Delta | Hidden Delta | Recurrent Delta | Birth Genome Unchanged? | Mean Policy Loss | Mean Value Loss | Mean Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `True` | 0.00000 | 0.00000 | 0.00000 |
| **PPO Only** | **0.025506** | **0.000918** | 0.000000 | 0.000000 | `True` | 0.00058 | 0.09347 | 1.38509 |
| **Evolution Only**| 0.000000 | 0.000000 | 0.000000 | 0.000000 | `True` | 0.00000 | 0.00000 | 0.00000 |
| **PPO + Evolution** | **0.025666** | **0.000893** | 0.000000 | 0.000000 | `True` | 0.00170 | 0.09370 | 1.38509 |

---

## 2. Prompt 3B: Long-Horizon Turnover & Cost Drag Diagnosis
* **Dataset & Protocol**: Frozen 1,800-session synthetic market (`market.csv`, SHA-256 `59daa371...`), 4 tradable assets (`SPY`, `AAA`, `BBB`, `CCC`), starting cash $10,000, 10 paired seeds (41–50).
* **Objective**: Identify why target-allocation PPO strategies chronically underperform static controls over multi-year simulation timelines.

### Primary Multi-Seed Summary (Test Split)
| Condition | Seeds | Validation NW ($) | Test NW ($) | Test Max Drawdown | Test Turnover | Mean PPO Updates | Mean Deaths | Mean Births |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** | 10 | 10,013.08 | **9,260.26** | 0.09377 | 17,461 | 0 | 1.0 | 0.0 |
| **PPO Only** | 10 | 10,011.79 | **9,267.50** | 0.09320 | **17,681** | 2,546.7 | 70.5 | 0.0 |
| **Evolution Only** | 10 | 10,011.09 | **9,258.11** | 0.09393 | 17,416 | 0 | 1.0 | 1.0 |
| **PPO + Evolution** | 10 | 10,009.70 | **9,274.80** | 0.09265 | **17,536** | 2,547.7 | 70.9 | 70.9 |

### Representative Cost & Turnover Breakdown (Seed 42 Tracing)
| Condition | Mean Turnover / Step | Cumulative Execution Cost ($) | Raw Log Return Sum | Mean Concentration | Mean Active Positions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | **40.4** | **$488** | **+0.081** | 0.252 | 4.0 |
| **PPO Only** | **6,256.9** | **$75,623** | **-8.202** | 0.397 | 4.0 |
| **PPO + Evolution** | **6,302.3** | **$76,172** | **-8.270** | 0.400 | 4.0 |
| **Conservative PPO** | 6,235.0 | $75,358 | -8.029 | 0.397 | 4.0 |
| **Rotated Asset Order**| 6,769.4 | $81,817 | -8.641 | 0.397 | 4.0 |

> [!WARNING]
> **Diagnostic Takeaway**: Target-portfolio PPO generated ~6,200 units of turnover per agent-day compared to ~40 for control, costing over $75,000 in simulated spread and slippage fees. This proved that transaction cost friction—not general calculation explosions—dominated PPO shortfall under early target-allocation schemes.

---

## 3. Prompt 4: Persistent Incremental Trade Interface Verification
* **Protocol Intervention**: Switched neural actions from full target-portfolio replacement to independent **Beta impulse trading** ($\Delta = 2u - 1$), scaling moves by a max trade fraction of net worth ($5\%$) against persistent portfolio holdings.

### Mechanical Comparison: Prompt 3B vs. Prompt 4 Action Architecture
| Condition | Prompt 3B Test NW ($) | Prompt 4 Test NW ($) | Prompt 3B Test Turnover | Prompt 4 Test Turnover | Turnover Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control** | 9,260.26 | **9,858.22** | 17,461 | **1,729.52** | **-90.1%** |
| **PPO Only** | 9,267.50 | **9,905.10** | 17,681 | **2,506.41** | **-85.8%** |
| **Evolution Only** | 9,258.11 | **9,851.87** | 17,416 | **1,528.48** | **-91.2%** |
| **PPO + Evolution** | 9,274.80 | **9,926.38** | 17,536 | **2,378.06** | **-86.4%** |

### Benchmark References on Prompt 4 Test Split
* **Cash**: `$9,865.50`
* **Random Portfolio**: `$8,940.47`
* **Equal Weight**: `$9,746.74`
* **Buy and Hold**: `$8,943.49`

---

## 4. Prompt 6: Rolling Market Senses & Sensor Ablation
* **Protocol**: Tested modular inclusion of rolling past-only market indicators (momentum, volatility, SMA trend) over 10 paired seeds on the Prompt 4 Beta impulse setup.

### Sensor Schema Comparison (Test Split Means across Seeds 41–50)
| Sensor Schema Condition | Input Dim | Validation NW ($) | Test NW ($) | Test Drawdown | Test Turnover | Win Rate vs. Default PPO (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Control** (No Learn) | 54 | 9,871.60 | 9,858.22 | 0.0192 | 1,729.52 | N/A |
| **Default PPO Baseline** | 54 | 9,824.46 | 9,905.10 | 0.0198 | 2,506.41 | Baseline |
| **Momentum Senses PPO** | 70 | 9,873.27 | **9,819.41** | 0.0254 | 1,860.92 | 2 / 10 Seeds (Overfit) |
| **Volatility/Trend Senses PPO**| 70 | 9,886.22 | **9,863.22** | 0.0228 | 1,797.88 | 4 / 10 Seeds (Overfit) |
| **Combined Senses PPO** | **98** | **9,902.97** | **9,989.73** | 0.0208 | 2,137.05 | **7 / 10 Seeds (+84.63 NW)** |
| **Combined PPO + Evolution** | **98** | **9,868.06** | **10,021.44** | **0.0194** | 2,427.73 | **7 / 10 Seeds (+116.34 NW)**|

---

## 5. Prompt 9: Comprehensive Walk-Forward Screening
* **Protocol**: Large-scale screening evaluation across **24 development markets**, **8 distinct market regime families**, and **3 walk-forward folds** (Seeds 41–45). Total evaluated simulation tasks: **2,520 evaluation runs** out of 3,240 planned jobs, completed with 0 unresolved lifecycle failures.

### Aggregate Walk-Forward Screening Results (Validation Range)
| Condition / Benchmark | Mean Return | Median Return | Profitable Run % | Win Rate vs Cash | Win Rate vs Equal Weight | Win Rate vs Momentum | Mean Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined PPO** | **-0.9174%** | -0.9157% | 12.78% | 56.11% | **42.78%** | 72.22% | 1.5452% |
| **Combined PPO + Evolution**| **-0.9255%** | -0.9515% | 13.61% | 54.44% | **42.78%** | 72.22% | 1.5490% |
| **Cash** | -0.9950% | -0.9950% | 0.00% | 0.00% | 43.06% | 72.22% | 0.9950% |
| **Equal Weight Benchmark** | **+2.2415%** | -0.1426% | **48.61%** | **56.94%** | **—** | **73.61%** | 9.1749% |
| **Buy & Hold Benchmark** | **+2.0108%** | -1.6830% | 43.06% | 47.22% | 50.00% | 70.83% | 11.9859% |
| **Momentum Benchmark** | -1.5346% | -5.5531% | 27.78% | 27.78% | 26.39% | — | 12.3691% |
| **Random Action Agent** | -5.2386% | -6.2038% | 10.28% | 12.22% | 5.28% | 38.89% | 9.7978% |

### Paired Screening Statistical Head-to-Head Comparisons
| Strategy Candidate | Comparator | Mean Return Difference | Median Difference | Wins / Losses / Ties (Across Folds) |
| :--- | :--- | :---: | :---: | :---: |
| **Combined PPO** | **Cash** | **+0.0776%** | +0.0793% | 202 / 158 / 0 |
| **Combined PPO** | **Equal Weight** | **-3.1588%** | -0.7466% | 154 / 206 / 0 |
| **Combined PPO** | **Momentum** | **+0.6173%** | +4.8141% | 260 / 100 / 0 |
| **PPO + Evolution** | **PPO Only** | **-0.0081%** | 0.0000% | 153 / 162 / 45 |

> [!NOTE]
> **Scientific Verdict**: Large-scale screening **did not establish general profitability** or a consistent alpha edge. Neural agents beat Momentum and Cash in most trials but underperformed simple Equal Weight portfolios (-3.16% mean return gap, losing in 206 of 360 paired walk-forward folds). Furthermore, evolutionary reproduction produced no significant outperformance over baseline PPO (-0.0081% difference). This demonstrates the strict intellectual rigor of Midas: positive short-horizon synthetic experiments do not automatically translate to genuine trading intelligence across complex out-of-sample regimes.
