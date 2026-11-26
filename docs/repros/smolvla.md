# SmolVLA — small VLA with a flow-matching action expert

> Reproduction record (LeRobot) · full training + finetune-from-base on self-collected
> SO-101 data · real-robot rollout · fp16 deployment within the 8 GB budget.
> Paper: SmolVLA (Hugging Face LeRobot team, 2025) · weights: `lerobot/smolvla_base`
> (Apache-2.0 code **and** weights) · run via `policy.type=smolvla`.

## 1. Background

SmolVLA is a ~450M-parameter VLA built for efficient training and on-robot inference:

- **SmolVLM2** vision-language backbone (0.5B class) — small enough to finetune on one
  GPU, strong enough to ground language in cluttered tabletop scenes.
- **Flow-matching action expert** (pi0-style): a small transformer expert predicts the
  *velocity* of an action-chunk flow; inference integrates the flow in ~10 steps.
- **Action chunking** with efficient streaming inference, designed for real-time control.

**How interviewers ask it:** "SmolVLA 和 pi0 什么关系？flow matching 怎么推导？
为什么动作头叫 action expert？微调时相机名对不上怎么办？" (§6).

## 2. Environment & data

Same stack as [act.md](act.md) (LeRobot + SO-101, dataset
`niuwengang123/lerobot_dataset_put_toy`, two cameras). SmolVLA adds no new dependencies —
it ships inside LeRobot.

## 3. Training recipes

### 3.1 Full training (from scratch)

```bash
lerobot-train \
  --dataset.repo_id=niuwengang123/lerobot_dataset_put_toy \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.revision=main \
  --dataset.streaming=false \
  --policy.type=smolvla \
  --output_dir=$OUTPUT_DIR/lerobot/output \
  --job_name=toy_smolvla \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=toy_project_smolvla \
  --policy.push_to_hub=false \
  --steps=40000 \
  --batch_size=8
```

### 3.2 Finetune from `lerobot/smolvla_base`

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=niuwengang123/lerobot_dataset_put_toy \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.revision=main \
  --dataset.streaming=false \
  --output_dir=$OUTPUT_DIR/lerobot/output \
  --job_name=toy_smolvla \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --wandb.enable=true \
  --wandb.project=toy_project_smolvla \
  --policy.push_to_hub=false \
  --steps=40000 \
  --batch_size=8
```

| Hyperparameter | Value | Note |
|---|---|---|
| steps | 40 000 | finetune converges much faster than scratch; earlier checkpoints already work |
| batch_size | 8 | ~<24 GB at this size (RTX 3090 runs it) |
| `--rename_map` | top→camera1, wrist→camera2 | the pretrained checkpoint expects `camera1/camera2` naming |
| `--policy.empty_cameras=1` | 1 | adds a missing-camera slot so the checkpoint's camera config matches |

**Why the rename_map exists (frequent "code-level" interview detail):** the base
checkpoint was pretrained with a fixed camera slot schema (`observation.images.camera1`,
`camera2`). Our dataset names cameras `top`/`wrist`. `--rename_map` remaps dataset keys
into the schema the checkpoint expects; `--policy.empty_cameras=1` keeps the config
consistent when the number of cameras differs. Forgetting these two flags is the most
common way a SmolVLA finetune silently underperforms.

## 4. Inference / real robot

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=g_follower_arm \
  --robot.cameras='{"wrist":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"fourcc":"MJPG"},"top":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --display_data=false \
  --task="Put the toy on the plate" \
  --duration=10000 \
  --policy.path=$OUTPUT_DIR/model/smolvla_040000/040000/pretrained_model
```

The same `--rename_map` must appear at rollout time — the policy config stored in the
checkpoint refers to `camera1/camera2`.

## 5. Results & retrospective

- Both recipes trained to convergence (wandb project `toy_project_smolvla`); the
  finetune run reaches usable behavior much earlier than the from-scratch run.
- Real-robot demo completes *"Put the toy on the plate"* (demo video recorded).
- **Deployment:** fp16 inference runs at ~1–2 GB VRAM — comfortable inside the 8 GB
  real-robot budget on an RTX 4060-class card. This is why SmolVLA is the deployment
  workhorse of this repo while larger models (pi0 class) are training-only.
- TODO (v0.1 backlog): formal 10-trial success-rate table
  ([real_eval_protocol.md](../experiments/real_eval_protocol.md)).

Pitfalls / lessons:

1. Camera-slot mismatch (see §3.2) — the #1 finetune footgun.
2. Finetune vs scratch: with ~40 episodes, finetuning from `smolvla_base` is clearly
   the right call; scratch training is only for studying the architecture.
3. Keep the `task` string at rollout identical to the dataset's `single_task` wording —
   the language channel is sensitive to phrasing.

## 6. Component breakdown (interview answers)

| Component | In SmolVLA | My one-line answer |
|---|---|---|
| VLM backbone | SmolVLM2 ~450M (image tokens + text) | "A compact VLM provides language grounding and scene understanding; it's finetuned, not pretrained here." |
| Action expert | Small transformer trained with **flow matching** | "A pi0-style expert: instead of denoising with DDPM, it regresses the velocity field v = x₁ − x₀ of a straight-path flow x_t = (1−t)x₀ + t·x₁; inference integrates ~10 Euler steps." |
| Chunking / execution | Predict action chunk, stream to the controller | "One flow integration per chunk; cheap enough for closed-loop replanning." |
| Loss | Flow-matching MSE on velocity | "E[‖v̂(x_t, t, obs) − (x₁ − x₀)‖²]; simpler than diffusion — no noise schedule tuning." |
| Config-level integration | `rename_map`, `empty_cameras` | "Pretrained checkpoints have camera-slot schemas; remap your dataset keys to match or performance silently drops." |
| Deployment | fp16 1–2 GB | "Fits an 8 GB card with room for cameras; quantization (int8/ONNX) is the next step, not a necessity." |

**"flow matching vs diffusion?"** — both learn a mapping from noise to data; diffusion
learns a noise-prediction objective over a stochastic schedule, flow matching regresses
a deterministic velocity field along straight paths. Flow matching usually needs fewer
inference steps and has fewer moving parts — which is why pi0/SmolVLA style experts use it.
