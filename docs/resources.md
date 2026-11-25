# Resources for small reasoning models and RLVR

Collected 2026-11-18. **Verification status:** GitHub repos marked ✓ were confirmed to exist with
`git ls-remote` on that date. Hugging Face dataset ids are from memory and search results and were
**not checked** (the Hub was unreachable from the machine that wrote this); confirm each id and its
licence on the Hub before use. Paper claims are summarised, not re-checked against the papers.

## Training data (verifiable answers)

| Dataset | Domain | Size (approx.) | Use it for | Hub id (unverified) |
|---|---|---|---|---|
| GSM8K | grade-school math | 7.5k train / 1.3k test | first runs, debugging rewards | `openai/gsm8k` |
| MATH (Hendrycks) | competition math, levels 1-5 | 7.5k train | the standard step up from GSM8K | `EleutherAI/hendrycks_math` or similar |
| DAPO-Math-17k | integer-answer competition math | 17k | the DAPO recipe's training set | `open-r1/DAPO-Math-17k-Processed` |
| DeepScaleR-Preview | AIME/AMC/Omni-Math mix | 40k | DeepScaleR-1.5B recipe | `agentica-org/DeepScaleR-Preview-Dataset` |
| Big-Math (RL-verified) | filtered math for RL | ~250k | large, cleaned pool | `SynthLabsAI/Big-Math-RL-Verified` |
| Skywork-OR1 RL data | math + code | ~100k | open reasoner recipe | `Skywork/Skywork-OR1-RL-Data` |
| Countdown | arithmetic puzzle | generated | TinyZero, cheapest R1-Zero repro | `Jiayi-Pan/Countdown-Tasks-3to4` |
| OpenR1-Math-220k | distilled R1 traces | 220k | SFT cold start / distillation | `open-r1/OpenR1-Math-220k` |
| OpenThoughts3 | distilled traces, math/code/science | 1.2M | SFT cold start / distillation | `open-thoughts/OpenThoughts3-1.2M` |
| rStar-Coder | competition code + tests | 418k problems | code RLVR | see paper (arXiv 2505.21297) |

## Procedural environments (unlimited, difficulty-controlled)

- ✓ [Reasoning Gym](https://github.com/open-thought/reasoning-gym): 100+ generators (algebra, logic,
  graphs, games, ARC-style grids) with verifiers and difficulty parameters. Paper: arXiv 2505.24760.
- ✓ [verifiers](https://github.com/PrimeIntellect-ai/verifiers) (Prime Intellect): environment +
  rubric abstraction, single- and multi-turn, used by the Environments Hub.
- SynLogic (35 logic tasks, adjustable difficulty), Enigmata (36 puzzle tasks with generators and
  verifiers, arXiv 2505.19914), Knights-and-Knaves logic puzzles.
- ✓ [ARC-AGI-3 agents](https://github.com/arcprize/ARC-AGI-3-Agents): API and agent templates for
  the interactive ARC-AGI-3 games; [technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).
- RAGEN and SPIRAL (multi-turn and self-play game environments), listed in awesome-RLVR.

## Evaluation benchmarks

| Benchmark | What it tests | Notes for small models |
|---|---|---|
| GSM8K test | basic multi-step arithmetic | saturates quickly; keep as a sanity check |
| MATH-500 | competition math subset | the main headline number at 0.5-1.7B |
| AMC23 (40 problems) | competition math | tiny: report mean over many samples |
| AIME 2024 / 2025 (30 each) | hard competition math | report avg@16 or pass@k, not one greedy run |
| OlympiadBench, Minerva Math | harder / science math | common in zero-RL papers' tables |
| GPQA-Diamond | graduate science MCQ | out-of-domain transfer check |
| LiveCodeBench, HumanEval+/MBPP+ | code | if you add code rewards |
| Reasoning Gym held-out tasks | procedural transfer | use different task families than training |
| ARC-AGI-1/2 (static), ARC-AGI-3 (interactive) | abstraction, agentic reasoning | project 5 |

Eval tooling: ✓ [lighteval](https://github.com/huggingface/lighteval) (used by open-r1),
lm-evaluation-harness, or a ~100-line vLLM + `math-verify` script (recommended: you control it).
Contamination: MATH/GSM8K are likely in pre-training data; AIME 2025 and procedural tasks are safer.

## Training frameworks

| Framework | Why use it here | Scale |
|---|---|---|
| ✓ [TRL](https://github.com/huggingface/trl) `GRPOTrainer` | simplest correct GRPO, vLLM rollouts, LoRA, accelerate | 1 → N GPUs; start here |
| ✓ [verl](https://github.com/verl-project/verl) | most-used research framework (DAPO, many recipes), Ray-based | multi-GPU; graduate to it |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | async FSDP2 RL, pairs with `verifiers` | multi-GPU |
| ✓ [rLLM](https://github.com/agentica-project/rllm) | agentic / multi-turn RL (DeepScaleR, DeepCoder authors) | multi-GPU |
| ART (OpenPipe), Atropos (Nous) | single-GPU-friendly agent RL | 1 GPU+ |
| OpenRLHF, AReaL, SkyRL, NeMo-RL, slime | larger-scale async systems | read, don't start here |

Survey of 16 libraries: [Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape) (HF, Mar 2026).

## Reproductions to copy (small-model, cheap)

- ✓ [TinyZero](https://github.com/Jiayi-Pan/TinyZero): R1-Zero on Countdown at 0.5-3B.
- ✓ [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason): zero-RL across 10+ base models.
- ✓ [Open-RS](https://github.com/knoveleng/open-rs): "RL for Reasoning in Small LLMs: What Works and What Doesn't" (arXiv 2503.16219), 1.5B on a small budget.
- ✓ [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero): Dr. GRPO.
- ✓ [open-r1](https://github.com/huggingface/open-r1): full R1 pipeline (SFT, GRPO, eval).
- Tina: LoRA RL for tiny reasoning models.

## Papers (read in this order)

1. DeepSeekMath (GRPO), arXiv 2402.03300
2. DeepSeek-R1, arXiv 2501.12948
3. Tulu 3 (RLVR term and recipe), arXiv 2411.15124
4. DAPO, arXiv 2503.14476
5. Understanding R1-Zero-like training (Dr. GRPO), from understand-r1-zero
6. Does RL really incentivise reasoning beyond the base model? ✓ [code](https://github.com/LeapLabTHU/limit-of-RLVR)
7. Spurious Rewards ✓ [code](https://github.com/ruixin31/Spurious_Rewards)
8. Reasoning Gym, arXiv 2505.24760
9. 100 Days After DeepSeek-R1 (replication survey), arXiv 2505.00551
10. Tufa Labs, [duck harness](https://tufalabs.ai/research/duck-harness/) (✓ [code](https://github.com/Tufalabs/duck-harness)) and
    Executable World Models, arXiv 2605.05138: harness design for interactive reasoning

## Distributed and efficient GPU training

Researched 2026-11-25. Links and course contents were read from each course's own page on that date;
prices and cohort dates were not shown where marked (unverified). In suggested order, with the
project each one supports:

| # | Resource | Cost | Covers | Use it for |
|---|---|---|---|---|
| 1 | [HF Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Feb 2025) | free | DP, ZeRO-1/2/3, tensor, sequence, context, pipeline and expert parallelism; memory and comms maths; scaling experiments | [extras/distributed](../projects/extras/distributed/): read alongside it; later torchtitan 2D/3D parallelism |
| 2 | [Stanford CS336](https://cs336.stanford.edu) (Spring 2026 videos on YouTube) | free online | assignment 2: profiling, Triton FlashAttention-2, memory-efficient distributed training loop | [extras/distributed](../projects/extras/distributed/) step 3 (from-scratch DDP); Triton/roofline work. Needs your own GPU time |
| 3 | [How to Scale Your Model](https://jax-ml.github.io/scaling-book) (Google DeepMind) | free | rooflines, sharding, FSDP, Megatron TP, pipelining, profiling, worked problems, JAX, a GPU chapter | performance reasoning for [01_serving](../projects/01_serving/) and distributed; a JAX/TPU port |
| 4 | [GPU MODE lectures](https://github.com/gpu-mode/lectures) | free | 100+ talks with code | pick by topic: 1 and 16 (profiling), 14 and 29 (Triton), 17 and 67 (NCCL), 39 (TorchTitan) |
| 5 | [PyTorch distributed tutorials](https://docs.pytorch.org/tutorials/distributed/home.html) + [torchtitan](https://github.com/pytorch/torchtitan) | free | FSDP2, DTensor, tensor parallel reference code | the code you actually write in [extras/distributed](../projects/extras/distributed/) and the RL projects' N-GPU runs |

**Paid option:** Zach Mueller's [Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale)
(Maven, live cohort with recordings). Implement DDP, ZeRO-2, context and expert parallelism from scratch,
with Slurm, Nsight and MFU profiling, on an included multi-node H100 cluster. Price and next cohort dates
not shown on the page (unverified). The only option here with real multi-node hardware.

**Deeper reading:** [CMU 15-779](https://www.cs.cmu.edu/~zhihaoj2/15-779/) (Fall 2025, LLM systems):
reading list on ZeRO/FSDP, Megatron, PipeDream, CUDA/Triton.

**Skipped:** NVIDIA DLI Model Parallelism workshop (only public date is from 2021, pricing on request);
MIT 6.5940 (mostly efficiency and TinyML, light on distributed training).

## Courses and lists

- [RLHF Book](https://rlhfbook.com/) (Lambert): policy gradients, RLVR chapters
- [HF LLM Course ch. 12](https://huggingface.co/learn/llm-course/en/chapter12/1): GRPO with TRL
- [Stanford CS336](https://stanford-cs336.github.io/spring2025/) assignment 5: SFT and RL for math reasoning
- [awesome-RLVR](https://github.com/opendilab/awesome-RLVR): curated, continually updated paper list
- [Prime Intellect Environments Hub](https://primeintellect.ai/blog/environments) and the
  [HF blog tour of it](https://huggingface.co/blog/anakin87/environments-hub)
