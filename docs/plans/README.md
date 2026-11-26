# Self-developed research track — plan (no code yet)

> Status: **plan only**. The classic-VLA reproductions in `docs/repros/` are the
> showcase deliverable; this document is the roadmap for the self-developed method
> (interview + research direction) that starts after v0.1.

## Direction: continual / lifelong learning for VLAs

**Why this direction**

- Interview differentiation: JDs for embodied-AI VLA roles consistently probe
  "世界模型 / 终身学习 / 自进化" as frontier topics; almost no candidate portfolio has a
  real continual-learning project on real hardware.
- Literature gap (verified in the 2025–2026 survey): per-task **LoRA expert
  accumulation** (CORAL/CLARE-style) and **selective replay** (Memory-Anchors-style)
  have not been coupled; and real-arm continual-learning benchmarks barely exist
  (one 10-task benchmark, arXiv 2605.26820, which also shows well-configured
  experience replay beats joint training under equal compute).
- Compute fit: LoRA on a SmolVLA-class base (<24 GB train) and base+experts at fp16
  (≤8 GB deploy) both fit the hard budgets.

**Method sketch (to be validated in MuJoCo first)**

1. **Per-task LoRA expert accumulation** — each new task trains a rank-16 LoRA expert
   on a frozen backbone; a small router selects/combines experts per task embedding.
2. **Novelty-gated replay** — a memory buffer stores anchor episodes; writes are gated
   by novelty/conflict scores (a new coupling — the research claim), replay trains the
   current expert against buffer + new data.
3. **Baselines behind the same `ContinualAdapter` interface** — joint training,
   well-tuned experience replay, LoRA-only; plus an in-context causal-memory arm
   (Zeva-style, zero-gradient) as a comparison.
4. **Metrics** — success per task over the stream, forgetting / backward transfer,
   forward transfer; OOD-instruction robustness as a secondary axis (motivated by the
   fresh negative result that replay alone leaves OOD drift, arXiv 2611.31511).

**Benchmark contribution** — a sequential-task protocol on MuJoCo SO-101 twin + real
arm (2–5 tasks × 30 episodes teleop), with fixed stream order and seeds. The protocol
itself is a deliverable (real-arm CL benchmarks are scarce).

## Supporting piece: from-scratch mini-VLA

A deliberately small VLA (ViT-tiny + small fusion transformer + flow-matching action
head + action chunking, ~10–25M params) implemented as a LeRobot `PreTrainedPolicy`
subclass. Purpose: (a) mastery of every component for interviews, (b) a cheap testbed
for continual-learning ablations before touching SmolVLA. Swappable action heads
(flow / diffusion / discrete tokens) give the "three heads, one trunk" interview demo.

## Deployment track (before the research track ships)

ONNX fp16/int8 export + latency/VRAM tables on the 8 GB x86 target; optional Zenoh
remote-inference split (policy server on cloud GPU). This doubles as the "deployment /
inference acceleration" interview story (the junior-vs-mid separator in JDs).

## Timeline (indicative)

| Phase | Content | Est. |
|---|---|---|
| v0.2 | mini-VLA + interview doc series | 1.5–2 weeks |
| v0.3 | continual-learning track + benchmark protocol | 2–3 weeks |
| v0.4 | deployment hardening (ONNX/int8, latency table) | 1 week |
| v0.5+ | pi0.5-LoRA LIBERO training record, blog series, extra baseline rows | continuous |

## Key references

- CORAL (arXiv 2603.09298), CLARE (arXiv 2601.09512) — LoRA expert accumulation
- Memory Anchors (arXiv 2608.26545) — selective replay
- Real-world 10-task continual-VLA benchmark (arXiv 2605.26820) — ER > joint training
- OOD drift despite near-zero forgetting (arXiv 2611.31511) — open problem
- Zeva / in-context causal learning (arXiv 2608.30880) — zero-gradient comparison arm
- InteractVLA (arXiv 2611.22584) — recursive self-improvement framing
