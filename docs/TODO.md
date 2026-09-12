# Midas Development Roadmap & Research TODOs

## PPO & Training Protocols

### 1. Randomized Episode Starting Sampling ($t_0$)
- **Objective:** Introduce randomized rollout start offsets across the historical training horizon rather than exclusively stepping sequentially from $t=0 \rightarrow t_{\text{end}}$.
- **Rationale:** Prevents policy gradient updates from over-indexing on the sequence of macro market regimes. Ensures every mini-batch contains a diverse mixture of bull, bear, and sideways market conditions sampled from random historical starting timestamps.
- **Architectural Requirements:**
  - Support arbitrary timestamp initialization for portfolio starting state.
  - Reset and reinitialize recurrent GRU hidden state buffers at sampled $t_0$ episode boundaries.
  - Preserve exact transactional cash and corporate action consistency across sampled windows.
