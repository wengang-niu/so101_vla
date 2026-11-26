# ACT — Action Chunking with Transformers

> Reproduction record (LeRobot implementation) · trained on self-collected SO-101 data ·
> real-robot rollout on the arm.
> Paper: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
> Hardware* (RSS 2023) · original code: tonyzhaozh/act (MIT) · this run uses LeRobot's
> `policy.type=act`.

## 1. Background

ACT turns a handful of low-cost teleop demonstrations into a precise manipulation
policy. Three ideas make it work:

- **Action chunking** — predict a short *sequence* of future actions (a "chunk") per
  inference instead of one action, which stabilizes long-horizon behavior.
- **CVAE over action sequences** — a variational latent captures task-level intent so
  the decoder does not average over multimodal demonstrations.
- **Temporal ensembling** — overlapping chunks executed back-to-back are combined with
  exponential weighting, smoothing jitter.

**How interviewers ask it:** "ACT 的原理是什么？为什么用 CVAE？action chunking 为什么
有效？时序集成怎么实现？" (see the component breakdown in §6 for ready answers).

## 2. Environment & data

```bash
conda create -y -n lerobot python=3.12 && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[feetech]"
```

Dataset: `niuwengang123/lerobot_dataset_put_toy` — task *"Put the toy on the plate"*,
~40 teleop episodes, two cameras (top + wrist, 640×480 @ 30 fps), LeRobotDataset v3.0.
Collection pipeline (calibration → teleop → record) is documented in
[../experiments/data_pipeline.md](../experiments/data_pipeline.md).

## 3. Training recipe

```bash
lerobot-train \
  --dataset.repo_id=niuwengang123/lerobot_dataset_put_toy \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.revision=main \
  --dataset.streaming=false \
  --policy.type=act \
  --output_dir=$OUTPUT_DIR/toy/act/ \
  --job_name=toy \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=toy_project_act \
  --policy.push_to_hub=false \
  --steps=20000 \
  --batch_size=6
```

| Hyperparameter | Value | Note |
|---|---|---|
| steps | 20 000 | loss plateaued well before the end on this dataset |
| batch_size | 6 | fits comfortably in <10 GB on an RTX 3090 |
| chunk size / KL weight / ensemble temp | LeRobot defaults | see `lerobot/policies/act/configuration_act.py` |
| optimizer / lr | LeRobot defaults | AdamW |

> GPU-mem: ACT is a small model (ResNet-18 backbones + a ~6-layer transformer); peak
> well under 10 GB at batch 6 — far inside the 48 GB training budget. Exact peak to be
> pinned with `nvitop` next run.

## 4. Inference / real robot

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=g_follower_arm \
  --robot.cameras='{"wrist":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30,"fourcc":"MJPG"},"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --display_data=true \
  --task="Put the toy on the plate" \
  --duration=10000 \
  --policy.path=$OUTPUT_DIR/toy/act/checkpoints/020000/pretrained_model
```

**Gotcha worth remembering (and mentioning in interviews):** camera device indices
(`/dev/video0` vs `/dev/video4`) are *not* stable across reboots/replugs — the mapping
in the training-time record command and the rollout command had to be swapped after a
replug. Always verify with `v4l2-ctl --list-devices` before a run, and keep the
`wrist`/`top` role mapping in one place.

## 5. Results & retrospective

- Training converged smoothly; wandb curves in project `toy_project_act`.
- Real-robot demo: policy repeatedly completes *"Put the toy on the plate"* from
  varying start poses (demo video recorded).
- TODO (v0.1 backlog): formal 10-trial success-rate table per
  [real_eval_protocol.md](../experiments/real_eval_protocol.md).

Pitfalls / lessons:

1. **Dataset quality dominates model tweaks** at 40 episodes — see LeRobot's
   [what makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)
   (consistent reset poses, clean background, full success coverage).
2. Chunk length trades smoothness vs reactivity: too long and the arm commits to stale
   plans; too short and behavior gets jittery. LeRobot defaults work well for SO-101.
3. Temporal ensembling is nearly free and visibly smooths trajectories — don't skip it.

## 6. Component breakdown (interview answers)

| Component | In ACT | My one-line answer |
|---|---|---|
| Perception | ResNet-18 per camera → 512-d features, concatenated with joint state | "Multi-view CNN features + proprioception, projected to the transformer width." |
| Sequence model | DETR-style transformer encoder-decoder | "Encoder consumes observations; decoder has learnable action queries that cross-attend to it." |
| Multimodality | CVAE: action-sequence encoder → latent z ~ (μ, σ); KL to N(0, I) | "The latent carries task-level intent so the decoder doesn't average conflicting demos; at inference we sample z = 0 (prior mode)." |
| Loss | L1 reconstruction + β·KL | "L1 on the action chunk, KL with a small weight (LeRobot default 10) to keep the latent informative but regularized." |
| Action chunking | Predict k future actions, execute, re-plan | "Fewer inference calls, smoother behavior, and it sidesteps compounding single-step errors." |
| Temporal ensembling | Weighted average over overlapping chunk predictions | "w_i ∝ exp(−0.01·i); blends the last few chunks' predictions for the current step — cheap and visibly smoother." |
| Inference loop | obs → chunk → execute k steps → repeat | "One forward per chunk; end-to-end latency is a non-issue at this model size." |

**"ACT vs Diffusion Policy?"** — ACT regresses the chunk directly (L1) with a CVAE for
multimodality; Diffusion Policy denoises the chunk with DDPM/DDIM, which handles
multimodality through the noise-prediction objective instead of an explicit latent.
ACT is simpler and trains faster; DP often gives more expressive action distributions.
