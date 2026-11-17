# Project 2: SFT then DPO, plus reward model vs LLM judge

**Status: stub. Nothing implemented or run.**

**Question:** what does DPO buy over SFT alone on a 0.5-1.5B model, and how well does a trained
reward model agree with a validated LLM judge? Where does each fail?

## Plan

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
`accelerate launch --config_file projects/03_distributed/configs/<ddp|fsdp>.yaml --num_processes N`.

## Files

- `configs/sft.yaml`, `configs/dpo.yaml`: starting hyperparameters (to tune)
- `train_sft.py`, `train_dpo.py`: stubs
