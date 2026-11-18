# Project 3: DDP vs FSDP2 vs DeepSpeed ZeRO-2/3 scaling benchmark

**Status: stub. Launcher configs written, training script not implemented, nothing run.**

**Question:** for the same 0.5-1.5B fine-tune, how do tokens/s, peak memory and MFU change across
strategies at 1, 2, 4 and 8 GPUs? When would you pick FSDP over ZeRO-3, and why?

## Plan

1. Write a minimal training loop in `train.py` (fixed synthetic or packed text batches, fixed
   steps) that logs tokens/s, `torch.cuda.max_memory_allocated()` and MFU
   (6 × params × tokens/s ÷ peak FLOPs; T4 fp16 peak is ~65 TFLOPs, unverified).
2. Run it under each launcher config below at 1 and 2 GPUs on Kaggle, then 4 and 8 on a rented node.
3. From scratch, following CS336 assignment 2: DDP with bucketed all-reduce, and a toy
   tensor-parallel MLP. Compare against `torch.nn.parallel.DistributedDataParallel`.
4. Produce the scaling table and a short write-up.

## 1 GPU → N GPUs

```bash
# 1 GPU
python projects/extras/distributed/train.py --config projects/extras/distributed/configs/train.yaml
# N GPUs, pick a strategy
accelerate launch --config_file projects/extras/distributed/configs/ddp.yaml   --num_processes N projects/extras/distributed/train.py --config ...
accelerate launch --config_file projects/extras/distributed/configs/fsdp.yaml  --num_processes N projects/extras/distributed/train.py --config ...
accelerate launch --config_file projects/extras/distributed/configs/zero3.yaml --num_processes N projects/extras/distributed/train.py --config ...
# or plain torchrun for the from-scratch DDP
torchrun --nproc_per_node N projects/extras/distributed/train.py --config ...
```

Check the accelerate config keys against the installed accelerate version (unverified);
`accelerate config` regenerates them interactively.
