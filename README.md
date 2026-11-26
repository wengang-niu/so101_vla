# so101_vla

**Classic VLA (vision-language-action) reproductions, documented for interviews — plus a
continual-learning VLA research track, trained under 48 GB GPU memory and deployed on a
self-built SO-101 arm with an 8 GB GPU budget.**

This repository has two goals:

1. **Classic VLA reproductions, recorded as showcase docs.** ACT, SmolVLA, pi0 — full
   training / inference recipes, real-robot rollouts on SO-101, results and lessons
   learned, each with a component breakdown written to answer interview questions.
   See [`docs/repros/`](docs/repros/).
2. **A self-developed method for interviews and research.** The innovation track targets
   **continual / lifelong learning for VLAs**: per-task LoRA expert accumulation +
   novelty-gated replay, benchmarked on a MuJoCo + real SO-101 sequential-task stream.
   A small from-scratch mini-VLA serves as the hands-on learning piece where every
   component (encoder → fusion → action head → inference loop) is written to be read.

Everything runs on the [LeRobot](https://github.com/huggingface/lerobot) stack, so the
**same code path** covers `lerobot-train` → `lerobot-eval` (PushT / Meta-World / LIBERO)
→ `lerobot-rollout` on the real arm.

## Status

Early development — see the roadmap. Design constraints are hard:

| Constraint | Value |
|---|---|
| Training budget | ≤ 48 GB GPU memory (local RTX 3090 + cloud GPU; portable recipes) |
| Real-robot inference | ≤ 8 GB GPU (x86 host + RTX 4060-class card) |
| Robot | self-built SO-101 (LeRobot ecosystem), wrist + top cameras |
| Simulation | MuJoCo (LeRobot SO-101 twin, LIBERO / Meta-World via `lerobot-eval`) |
| License | Apache-2.0 (code). No restricted-weight redistribution — see below. |

## Reproduction showcase

| Algorithm | What is documented | Doc |
|---|---|---|
| **ACT** | train + real-robot rollout on SO-101 ("Put the toy on the plate"), CVAE/chunking breakdown | [docs/repros/act.md](docs/repros/act.md) |
| **SmolVLA** | full-train + finetune-from-base recipes, camera remapping, fp16 deployment numbers | [docs/repros/smolvla.md](docs/repros/smolvla.md) |
| **pi0** | full-train + LoRA finetune recipes from `lerobot/pi0_base` | [docs/repros/pi0.md](docs/repros/pi0.md) |

Data-collection and hardware setup (calibration, teleop, LeRobotDataset v3.0, time
alignment) are recorded in [docs/experiments/data_pipeline.md](docs/experiments/data_pipeline.md).
The real-robot evaluation protocol is in
[docs/experiments/real_eval_protocol.md](docs/experiments/real_eval_protocol.md).

## Roadmap

| Version | Theme |
|---|---|
| **v0.1** | Scaffolding + classic-repro showcase docs + data-pipeline record + real-eval protocol |
| **v0.2** | From-scratch mini-VLA (flow-matching head) + interview doc series |
| **v0.3** | Continual-learning track: LoRA-expert accumulation + gated replay + real-arm CL benchmark |
| **v0.4** | Deployment hardening: ONNX fp16/int8, latency table, Zenoh remote inference |
| v0.5+ | pi0.5-LoRA training record (LIBERO parity band), blog series, optional baseline rows |

## Repository layout

```
docs/
  repros/            # ★ classic-VLA showcase records (recipes, results, interview breakdowns)
  experiments/       # data pipeline, real-eval protocol, experiment logs
  interview/         # interview-question docs ("build a VLA from scratch" is the flagship)
  architecture.md    # design rules
src/so101_vla/
  policies/          # from-scratch mini-VLA (LeRobot PreTrainedPolicy subclass)
  modules/           # building blocks: vision, language, fusion, action heads, temporal
  continual/         # innovation track: ContinualAdapter interface + strategies
  integrations/      # thin LeRobot wrappers (no copied code)
  deployment/        # ONNX export, quantization, latency benchmarking
configs/train/       # one YAML per recipe, each with a # GPU-mem budget header
scripts/             # eval / rollout wrappers
benchmarks/          # result summaries (JSONL + markdown; never weights)
tests/               # unit tests + secret scanner
```

## Getting started

```bash
# 1. Environment (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Smoke tests (no GPU required)
pytest

# 3. Read the reproductions
open docs/repros/act.md

# 4. Real SO-101 rollout with success logging (after training a policy)
bash scripts/eval_real_so101.sh <policy>
```

Training hardware here: one RTX 3090 (24 GB) locally plus cloud GPUs for larger runs.

## License & third-party weights

Code in this repository is **Apache-2.0**. We do not vendor code or weights from
projects with restrictive licenses (openpi / Gemma Terms of Use, OpenVLA / Llama-2
community license, GR00T / NVIDIA Open Model License, Zeva / OpenMDW). Where such models
are used (e.g. pi0 recipes via LeRobot), they run under their own terms and weights are
never redistributed here.

## Security

Never commit credentials. This repo scans for secrets in CI
(`tests/test_no_secrets.py` + gitleaks pre-commit). Use `hf auth login` and environment
variables for Hugging Face / W&B access — all commands in the docs use placeholders.
