# Architecture & design rules

`so101_vla` is organized so that **one code path** covers training, simulation
evaluation, and real-robot deployment, and so that every line of policy code is
explainable in an interview.

## The one-code-path rule

Every policy is a LeRobot `PreTrainedPolicy` subclass (LeRobot 0.6.x API:
`PreTrainedConfig` + pre/post processors). Consequences:

- `lerobot-train --policy.type=<name>` trains anything in this repo (registered via
  `src/so101_vla/integrations/lerobot_plugin.py`).
- `lerobot-eval` / `lerobot-rollout` consume the same checkpoints — PushT, Meta-World,
  LIBERO, the MuJoCo SO-101 twin, and the real arm all reuse one inference contract
  (`predict_action_chunk` / `select_action`).
- New policies plug in as config + model files; no bespoke training loops.

## Two code zones, hard-separated

| Zone | What lives there | Rule |
|---|---|---|
| `src/so101_vla/policies/` + `src/so101_vla/modules/` + `src/so101_vla/continual/` | Self-written code (from-scratch mini-VLA, continual-learning track) | Never copy code from restrictively-licensed projects (openpi / Gemma ToU, OpenVLA / Llama-2, GR00T / NVIDIA OML, Zeva / OpenMDW). Read the paper, write the math. |
| `src/so101_vla/integrations/` | Thin wrappers over Apache-2.0 LeRobot / SmolVLA | Call into `lerobot.*`; never paste their implementation code into this repo. |
| `docs/repros/` | Classic-VLA **showcase records** (recipes, results, breakdowns) | Documentation only — classic algorithms are not reimplemented in this repo. |

Review checklist for every policy PR:

1. Is the implementation traceable to the paper (module-by-module mapping in
   `docs/repros/<name>.md`)?
2. Any snippet that looks copied from a restricted repo? (Reject.)
3. Any dependency outside Apache-2.0 / MIT / BSD? (Reject; `pip-licenses` in CI.)
4. Any hardcoded path, token, or key? (Reject; `tests/test_no_secrets.py`.)

## Budget discipline

Every training config in `configs/train/` starts with a header comment:

```yaml
# GPU-mem: <peak>GB @ batch=<b>; fits: 48GB-train / 8GB-infer
```

Recipes that exceed the 48 GB training budget or the 8 GB deployment budget are
rejected at review time. Local development runs on a single RTX 3090 (24 GB);
larger runs go to cloud GPUs — configs must stay portable (no absolute paths).

## Innovation slot: an interface, not a module

The continual-learning track plugs in behind one ABC
(`src/so101_vla/continual/interface.py::ContinualAdapter`):

- `lifelong_lora` — per-task LoRA expert accumulation + automatic routing (default)
- `er_replay` — experience replay baseline (tuned carefully; under equal compute a
  well-configured ER beats joint training)
- `joint` — joint-training baseline
- `iccl_memory` — frozen-policy in-context causal memory (Zeva-style), as a
  zero-gradient comparison arm

Policy code never imports a concrete strategy; configs select one. This keeps the
innovation track swappable and keeps baselines honest (same interface, same eval).

## Module map (from-scratch mini-VLA, v0.2)

```
observations ─► vision encoder (ViT-tiny, multi-camera tokens)
task text    ─► language encoder (small transformer, fixed-length prefix)
                     │
                     ▼
              fusion (action-query tokens cross-attend to vision+language)
                     │
                     ▼
              action head ── flow_matching | diffusion | discrete_tokens
                     │
                     ▼
              action chunk (k steps) + temporal ensembling / RTC
```

Each component maps to interview questions in `docs/interview/`.

## Real-robot evaluation protocol

See `docs/experiments/real_eval_protocol.md`. Summary: fixed object poses, 60 s
timeout, 10 trials per policy, success = task-defined terminal condition, results
logged as JSONL in `benchmarks/results/` and summarized in `benchmarks/RESULTS.md`.
