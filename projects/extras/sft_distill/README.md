# Extra: SFT warm-start / distillation, then optional DPO

**Status: stub. Nothing implemented or run.**

**Question:** what does DPO buy over SFT alone on a 0.5-1.5B model, and how well does a trained
reward model agree with a validated LLM judge? Where does each fail?

## Role in the reasoning roadmap

R1-style pipelines often start RL from a model SFT'd on long reasoning traces (cold start).
The reasoning version of this project: SFT a 0.5-1.5B model on a subset of distilled traces
(e.g. OpenR1-Math or OpenThoughts, see docs/resources.md), then compare GRPO from base vs from
the SFT checkpoint in project 2. The DPO plan below is kept as an optional preference-tuning track.

## Original plan (preference tuning)

1. **SFT** Qwen2.5-0.5B (base) on a subset of the Tulu 3 SFT mix or UltraChat with TRL `SFTTrainer`.
   Start with LoRA on a T4, full fine-tune on a rented A100/H100.
2. **DPO** from the SFT checkpoint on UltraFeedback (binarized) with TRL `DPOTrainer`.
   Sweep `beta` in {0.05, 0.1, 0.3}.
3. **Reward model** on the same preferences (TRL `RewardTrainer`), scored on RewardBench.
4. **RM vs judge:** score held-out pairs with the RM and an LLM judge; report agreement and a
   failure-case taxonomy (length bias, refusals, formatting).
5. **Eval:** IFEval and AlpacaEval 2 (length-controlled) or MT-Bench for base / SFT / DPO.

Dataset ids and licences to confirm on the Hub before use (unverified):
`allenai/tulu-3-sft-mixture`, `HuggingFaceH4/ultrachat_200k`, `HuggingFaceH4/ultrafeedback_binarized`,
`allenai/reward-bench`.

## Scaling 1 → N GPUs

The same script runs on 1 GPU with `python` and on N with
`accelerate launch --config_file projects/extras/distributed/configs/<ddp|fsdp>.yaml --num_processes N`.

## Files

- `configs/sft.yaml`, `configs/dpo.yaml`: starting hyperparameters (to tune)
- `train_sft.py`, `train_dpo.py`: stubs
