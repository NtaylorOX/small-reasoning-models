# Project 3: What does RLVR actually do? Ablations on small models

**Status: plan only.** Builds on project 2's baseline and eval script.

These are the questions frontier labs and recent papers argue about, and each is cheap enough to
test at 0.5-1.7B. Pick two or three, not all.

| Question | Experiment | Starting point |
|---|---|---|
| Does RL add capability or only sharpen what the base model can already sample? | pass@k curves, k = 1…256, base vs RL model; where do they cross? | [limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR) |
| Is the reward doing the work? | correct vs random vs format-only vs incorrect rewards, Qwen vs non-Qwen model | [Spurious Rewards](https://github.com/ruixin31/Spurious_Rewards) |
| Which GRPO fixes matter at small scale? | GRPO vs Dr. GRPO vs DAPO tricks (clip-higher, dynamic sampling, token-level loss, overlong shaping) | [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero), DAPO paper |
| How much does group size / KL / temperature matter? | G ∈ {4, 8, 16}, β ∈ {0, 0.04}, T ∈ {0.6, 1.0} at a fixed compute budget | project 2 config |
| Length vs accuracy | does response length growth track accuracy, or is it a loss-normalisation artefact? | Dr. GRPO analysis |
| Can a tiny model learn from 1 example? | 1-shot / few-shot RLVR on one MATH problem | One-shot RLVR paper (look up before use) |

Every claim needs at least 3 seeds and the eval script from project 2. Write it up as a short
preprint-style report: this is the most research-shaped deliverable in the repo.
