# Project 4: Procedural reasoning environments and curricula

**Status: plan only.**

Math datasets are finite and contaminated. Procedural generators give unlimited, difficulty-
controlled, automatically verifiable tasks, which is where RLVR research is moving.

**Questions:** does training on a curriculum of logic/puzzle tasks transfer to held-out tasks and
to math? Does an adaptive curriculum (keep tasks near 50% success) beat a fixed mix?

## Plan

1. Use [Reasoning Gym](https://github.com/open-thought/reasoning-gym) (100+ generators with
   verifiers and difficulty knobs) and/or [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
   environments, which plug into TRL-style trainers and Prime Intellect's Environments Hub.
2. Pick a train family (e.g. arithmetic + logic) and held-out families (e.g. graphs, ARC-style grids).
3. Compare fixed difficulty, linear ramp, and success-rate-adaptive curricula at equal compute.
4. Report in-distribution, held-out-task and MATH-500 transfer.

Related task suites to consider (check licences first): SynLogic, Enigmata, Knights-and-Knaves,
Countdown. See [docs/resources.md](../../docs/resources.md).
