# Parallel track: agent security, red teaming and safety evals

Researched 2026-11-28. Runs alongside the RLVR roadmap, not instead of it: one security project at a
time, interleaved with the main projects. The hands-on code lives in a separate sandbox repo
(proposed name `agent-security-lab`, see [the last section](#the-sandbox-repo)); this file is the plan
and reading list.

**Verification status.** GitHub repos marked ✓ were confirmed to exist with `git ls-remote` on
2026-11-28. arXiv ids for older papers are from memory; ids for 2026 papers come from search results
on that date. Course prices and contents are as shown on their pages or **(unverified)**. Check
dataset and benchmark licences before use: several harm datasets are research-only.

## Ground rules for this track

This is defensive research: measuring how robust safety training and agent defences are, on
models and systems you own, in an isolated sandbox. The rules are written to keep it that way.

1. **Local targets only.** Attacks run against local open-weight models and toy services inside the
   sandbox. No third-party APIs, websites or systems as targets. Hosted models are used only as
   judges or as baselines through their normal terms of use.
2. **Measure with canaries, not harm.** Agent attacks succeed by touching a planted canary (a fake
   secret, a `*.test` address, a forbidden tool call). Success is decided by deterministic code, not
   by reading harmful text.
3. **Scores in git, completions out of git.** Jailbreak and refusal experiments commit judge scores,
   refusal rates and prompt ids. Raw model outputs on harmful-behaviour prompts stay in a git-ignored
   `results/raw/` and are not published.
4. **No released uncensored weights.** Refusal-direction work studies how robust refusal is and how to
   make it more robust. Ablated models stay local; publish the analysis, not the weights.
5. **Network egress off by default** in the sandbox (see the repo's threat model).

## What the field looks like (Nov 2026)

- **Automated red teaming** has settled into three families: optimisation over tokens (GCG and its
  descendants), attacker-LLM search (PAIR, TAP, Rainbow Teaming) and simple adaptive templates plus
  random search, which Andriushchenko et al. showed reaches near-100% attack success on many
  aligned models. Standard practice is to report attack success rate (ASR) on a fixed behaviour set
  (HarmBench, JailbreakBench) with a fixed judge (HarmBench classifier, StrongREJECT), as a function
  of query budget.
- **Refusal is shallow.** Arditi et al. found refusal is largely mediated by a single direction in
  the residual stream; projecting it out ("abliteration") removes refusal while mostly keeping
  capabilities. Qi et al. showed safety alignment is concentrated in the first few output tokens and
  that small fine-tunes erase it. Defences now aim at depth: circuit breakers / representation
  rerouting, tamper-resistant training (TAR), and fine-tuning on extended refusals so the direction is
  no longer one-dimensional (arXiv 2505.19056). Whether these hold up is an open, measurable question.
- **Prompt injection is the agent problem.** Indirect injection through tool outputs (emails, web
  pages, files) hijacks tool-using agents. AgentDojo is the reference benchmark (utility and ASR
  together); 2026 work moves to dynamic and generated task suites (AgentDyn arXiv 2602.03117,
  AutoDojo arXiv 2606.15057) and automated injection search (PI-Hunter arXiv 2606.12737). Defences
  split into model-level (StruQ, SecAlign / Meta SecAlign), prompt-level (spotlighting) and
  system-level (CaMeL: separate control flow from untrusted data, capability tracking). System-level
  designs are the only ones with guarantees.
- **Sandboxes are part of the eval.** UK AISI's Inspect has a sandboxing toolkit (Docker,
  Kubernetes, Proxmox) and a container-escape benchmark (arXiv 2603.02277): agents that run code need
  isolation that is itself tested.

## Courses and hands-on practice

In suggested order:

| # | Resource | Cost | Covers | Use it for |
|---|---|---|---|---|
| 1 | [Microsoft AI Red Teaming training](https://learn.microsoft.com/en-us/security/ai-red-team/training) + ✓ [AI-Red-Teaming-Playground-Labs](https://github.com/microsoft/AI-Red-Teaming-Playground-Labs) | free | red-team methodology, direct and indirect prompt injection, labs in Docker | vocabulary and the "how real AI red teams work" picture. Pair with [Lessons from red teaming 100 generative AI products](https://arxiv.org/abs/2501.07238) |
| 2 | [XLab AI Security Guide](https://www.lesswrong.com/posts/95DXi2Wivs4v65evz/introducing-the-xlab-ai-security-guide) (UChicago) | free | coding exercises: jailbreaks, GCG, adversarial robustness, defences | S2 and S3: implement attacks yourself before using libraries |
| 3 | ✓ [ARENA](https://github.com/callummcdougall/ARENA_3.0) chapter 1 (interpretability) | free | TransformerLens, activation patching, steering | the tooling for S3 (refusal direction) |
| 4 | [Inspect docs](https://inspect.aisi.org.uk/) + ✓ [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) | free | tasks, solvers, scorers, sandboxes, agent evals | the eval harness for every project here |
| 5 | Lakera Gandalf, HackAPrompt, Dreadnode Crucible | free | prompt-injection and ML-security CTFs | a few evenings of intuition; not CV material |
| 6 | [BlueDot Impact](https://bluedot.org/) AI safety courses | free, cohort-based | alignment and governance context | optional framing for interviews |

**Reference frameworks:** OWASP Top 10 for LLM Applications (2025) and its agentic-AI threat list,
MITRE ATLAS, NIST AI 600-1. Skim once; useful for threat models and job interviews.

**Paid options (unverified prices):** Learn Prompting's AI Red Teaming masterclass (Maven), Udacity's
AI Security Engineer nanodegree. Not needed for this plan; the free material above is deeper.

## Tools

| Tool | What it is | Where it fits |
|---|---|---|
| ✓ [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) / ✓ [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) / ✓ [inspect_cyber](https://github.com/UKGovernmentBEIS/inspect_cyber) | UK AISI eval framework, sandboxes, ready evals (AgentHarm, StrongREJECT, Cybench, WMDP...) | default harness; matches triage-gym |
| ✓ [garak](https://github.com/NVIDIA/garak) | LLM vulnerability scanner, many probes | S1: broad scan of a local model in an afternoon |
| ✓ [PyRIT](https://github.com/Azure/PyRIT) | Microsoft's red-teaming orchestration (multi-turn, converters, scorers) | S2: orchestrating attacker/target/judge loops |
| ✓ [promptfoo](https://github.com/promptfoo/promptfoo) | config-driven red-team and regression tests | quick app-level regression suites |
| ✓ [HarmBench](https://github.com/centerforaisafety/HarmBench) | behaviours, attack implementations, classifier judge | S2 benchmark and judge |
| ✓ [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) | 100 behaviours, leaderboard, artifacts | S2 benchmark, smaller and cheaper |
| ✓ [StrongREJECT](https://github.com/alexandrasouly/strongreject) | rubric judge that penalises empty "jailbreaks" | S2/S3 judge |
| ✓ [nanoGCG](https://github.com/GraySwanAI/nanoGCG), ✓ [llm-attacks](https://github.com/llm-attacks/llm-attacks) | GCG implementations | S2 white-box baseline |
| ✓ [PAIR](https://github.com/patrickrchao/JailbreakingLLMs), ✓ [TAP](https://github.com/RICommunity/TAP), ✓ [adaptive attacks](https://github.com/tml-epfl/llm-adaptive-attacks) | black-box attacker-LLM and random-search attacks | S2 |
| ✓ [refusal_direction](https://github.com/andyrdt/refusal_direction), ✓ [abliterator](https://github.com/FailSpy/abliterator), ✓ [heretic](https://github.com/p-e-w/heretic) | refusal-direction extraction and ablation | S3 (measurement only, see ground rules) |
| ✓ [circuit-breakers](https://github.com/GraySwanAI/circuit-breakers) | representation-rerouting defence | S3 defence baseline |
| ✓ [AgentDojo](https://github.com/ethz-spylab/agentdojo) | prompt-injection benchmark for tool agents, utility + ASR | S4 reference benchmark |
| ✓ [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent), ✓ [BIPIA](https://github.com/microsoft/BIPIA), ✓ [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench), ✓ [OS-Harm](https://github.com/tml-epfl/os-harm) | more agent-injection and agent-safety benchmarks | S4 extra suites |
| ✓ [CaMeL](https://github.com/google-research/camel-prompt-injection), ✓ [StruQ](https://github.com/Sizhe-Chen/StruQ), ✓ [Meta SecAlign](https://github.com/facebookresearch/Meta_SecAlign) | injection defences: system-level and fine-tuned | S4 defences |
| ✓ [PurpleLlama](https://github.com/meta-llama/PurpleLlama) | CyberSecEval, Llama Guard, Prompt Guard | S4 guard baselines, S6 cyber evals |
| ✓ [Cybench](https://github.com/andyzorigin/cybench) | 40 professional CTF tasks for agents | S6 |
| ✓ [ControlArena](https://github.com/UKGovernmentBEIS/control-arena) | AI control settings (untrusted agent, monitors) | stretch: monitoring research |
| ✓ [mcp-scan](https://github.com/invariantlabs-ai/mcp-scan) | scans MCP servers for tool poisoning | S4: MCP tool-description injection |
| Isolation: ✓ [gVisor](https://github.com/google/gvisor), ✓ [Firecracker](https://github.com/firecracker-microvm/firecracker), ✓ [Kata](https://github.com/kata-containers/kata-containers) | stronger sandboxes than plain Docker | sandbox runtime options |

## Papers (read in this order)

1. Perez et al., Red Teaming Language Models with Language Models (arXiv 2202.03286)
2. Zou et al., GCG: Universal and Transferable Adversarial Attacks (arXiv 2307.15043)
3. Chao et al., PAIR (arXiv 2310.08419); Mehrotra et al., TAP (arXiv 2312.02119)
4. Mazeika et al., HarmBench (arXiv 2402.04249); Souly et al., StrongREJECT (arXiv 2402.10260)
5. Andriushchenko et al., Jailbreaking leading safety-aligned LLMs with simple adaptive attacks (arXiv 2404.02151)
6. Arditi et al., Refusal in LLMs is mediated by a single direction (arXiv 2406.11717)
7. Qi et al., Fine-tuning aligned models compromises safety (arXiv 2310.03693) and Safety alignment should be more than a few tokens deep (arXiv 2406.05946)
8. Zou et al., Circuit breakers (arXiv 2406.04313); Tamirisa et al., Tamper-resistant safeguards (arXiv 2408.00761); An embarrassingly simple defense against abliteration (arXiv 2505.19056)
9. Greshake et al., Indirect prompt injection (arXiv 2302.12173)
10. Debenedetti et al., AgentDojo (arXiv 2406.13352); Zhan et al., InjecAgent (arXiv 2403.02691); Andriushchenko et al., AgentHarm (arXiv 2410.09024)
11. Defences: Hines et al., Spotlighting (arXiv 2403.14720); Chen et al., StruQ (arXiv 2402.06363), SecAlign (arXiv 2410.05451); Debenedetti et al., CaMeL (arXiv 2503.18813)
12. 2026: AgentDyn (arXiv 2602.03117), AutoDojo (arXiv 2606.15057), PI-Hunter (arXiv 2606.12737), container sandbox escape (arXiv 2603.02277)
13. Stretch: Hubinger et al., Sleeper agents (arXiv 2401.05566); Zhang et al., Cybench (arXiv 2408.08926)

## Projects

All run on local models (Qwen 2.5/3 at 0.5-8B, Llama 3.x 8B) served by vLLM, so the serving work
from project 1 carries over. S1-S4 fit on Kaggle's 2x T4 or one rented GPU.

| # | Project | Output | Compute | Bridges to |
|---|---|---|---|---|
| S1 | **Eval literacy.** garak scan plus StrongREJECT and HarmBench refusal rates for 3-4 small models, run through Inspect | a reproducible safety-eval table per model | T4 | project 1 (vLLM serving) |
| S2 | **Automated red teaming.** Implement PAIR and the adaptive random-search attack yourself (local attacker, local target), compare with nanoGCG; JailbreakBench behaviours, StrongREJECT judge | ASR vs query-budget curves, attack cost per success | T4 / 1x A100 | eval statistics in `src/lab/stats.py` |
| S3 | **How deep is refusal?** Reproduce the refusal direction on 2-3 models; measure refusal rate vs capability (MMLU, GSM8K) under ablation; then test defences (extended-refusal fine-tune, circuit breakers) and ask whether **RLVR training from project 2 erodes refusal** | short preprint-style report; no ablated weights released | T4 for 1.5B, A100 for 8B | project 2/3 checkpoints: a novel, cheap question |
| S4 | **Prompt injection against tool agents.** AgentDojo with local models first; then the sandbox repo's own canary suite; compare no defence, spotlighting, a guard model and a CaMeL-style planner/executor split | utility vs ASR Pareto plot per defence | T4 | project 5 (harness agents), triage-gym tools |
| S5 | **RL for red teaming and robustness.** Train a small attacker with GRPO to find injections against a fixed defended agent (reward = canary hit in the sandbox), then adversarially train the defender on what it finds | does RL find attacks templates miss; does adversarial training transfer to AgentDojo | 1-2x A100 | the RLVR infrastructure from projects 2-4 |
| S6 | **Cyber capability evals.** Cybench / inspect_cyber CTFs with local models in the sandbox; measure solve rate vs model size and harness | capability table, sandbox hardening notes | T4 / A100 | project 5 harness |

S3 and S5 are the two that combine both tracks into something new: "does RLVR on math change
safety behaviour?" and "RL-trained attackers against agent defences". Both are measurable with
existing benchmarks, which fits the "known-good datasets first" rule.

## The sandbox repo

A separate repo keeps attack code, containers and raw logs away from the training repo. Proposed:
`agent-security-lab`, private:

- `sandbox/`: Docker Compose stack on an internal-only network (no internet), optional gVisor
  (`runsc`) runtime; a local vLLM/Ollama model server, a tool-using agent container, and mock
  services (mail, files, web) seeded with canary secrets.
- `src/agentsec/`: a pure-Python simulated workspace (tools, injection points), the agent loop behind
  a small `Model` interface, defences as swappable wrappers, deterministic scoring (utility, ASR,
  Wilson intervals), and refusal-direction measurement helpers.
- Inspect tasks wrapping the same scenarios so results come out as Inspect logs.
- `docs/threat-model.md`: what the sandbox does and does not protect against, and the ground rules
  above.
