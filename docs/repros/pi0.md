# pi0 — flow-matching action expert on a VLM backbone

> Reproduction record (LeRobot `policy.type=pi0`) · full training + LoRA finetune from
> `lerobot/pi0_base` on self-collected SO-101 data · inference pending.
> Paper: Black et al., *π0: A Vision-Language-Action Flow Model for General Robot
> Control* (Physical Intelligence, 2024) · weights via LeRobot under Gemma Terms of Use
> (weights are **not** redistributed in this repo).

## 1. Background

pi0 pairs a PaliGemma-class VLM (≈3B) with a lightweight **action expert** (≈300M) that
generates action chunks via **flow matching**, trained as a prefix-LM where the expert
attends to VLM features. It is the strongest open recipe at the "large VLA" tier that a
single-node budget can still finetune — with memory tricks. pi0.5 (the follow-up,
covered later on this track) improves long-horizon / language-grounded behavior on the
same architecture family.

**How interviewers ask it:** "pi0 和 SmolVLA 的区别？action expert 为什么单独训练？
LoRA 微调 VLA 怎么配？全量微调 48GB 怎么塞进去？" (§6).

## 2. Environment & data

```bash
pip install -e ".[pi]"        # pi0 dependencies in LeRobot
pip install 'lerobot[dataset]'
pip install 'lerobot[training]'
pip install 'lerobot[peft]'   # LoRA finetune only
```

Same SO-101 dataset as [act.md](act.md). Training ran on a cloud GPU node
(48 GB-class); LoRA finetune also runs on a single RTX 3090.

## 3. Training recipes

### 3.1 Full training (with memory tricks)

```bash
lerobot-train \
  --dataset.repo_id=niuwengang123/lerobot_dataset_put_toy \
  --dataset.revision=main \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.streaming=false \
  --policy.type=pi0 \
  --output_dir=$OUTPUT_DIR/toy_pi0 \
  --job_name=toy_pi0 \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=50000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=toy_project_pi_0 \
  --batch_size=8
```

### 3.2 LoRA finetune

```bash
lerobot-train \
  --dataset.repo_id=niuwengang123/lerobot_dataset_put_toy \
  --dataset.revision=main \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.streaming=false \
  --policy.type=pi0 \
  --output_dir=$OUTPUT_DIR/toy_pi0_lora \
  --job_name=toy_pi0_lora \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=3000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project=toy_project_pi_0 \
  --batch_size=8 \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16
```

| Hyperparameter | Value | Why it matters |
|---|---|---|
| `gradient_checkpointing=true` | both | recomputes activations in the backward pass — the single biggest memory lever (~2–3× savings) |
| `dtype=bfloat16` | both | halves optimizer/activation memory vs fp32; bf16 (not fp16) avoids overflow in the VLM |
| `compile_model=true` | full only | `torch.compile` kernels; disabled for LoRA because compile + PEFT hooks interact badly |
| `freeze_vision_encoder=false` | full | full finetune updates the vision tower; freezing it is the lighter alternative |
| `train_expert_only=false` | full | trains VLM + expert; `true` = expert-only is the cheap mode |
| `peft.r=16 / lora_alpha=16` | LoRA | rank-16 adapters on attention projections; ~1% of trainable params |
| steps | 50 000 / 3 000 | LoRA converges in a fraction of the steps |

> GPU-mem: full finetune at batch 8 with the tricks above fits a 48 GB node (measured
> peak to be pinned from the run logs); LoRA finetune peaks well below 24 GB and runs
> on the RTX 3090. This recipe suite is the repo's "how to fit a 3B VLA into ≤48 GB"
> interview answer.

## 4. Inference / real robot — **TODO (recorded gap)**

Rollout on SO-101 is **not yet recorded** for pi0 (known gap in my notes). When filled
in, this section will contain the `lerobot-rollout` command and results. Constraints to
keep in mind meanwhile:

- pi0 bf16 inference runs **>8 GB VRAM** — it does not fit the 8 GB real-robot budget
  out of the box. Options: int8 quantization, ONNX/TensorRT export, or LeRobot's
  remote-inference split (policy server on cloud GPU, arm-side client). All three are
  planned on the deployment track.
- The rollout must use the same dataset-rename conventions as training if cameras are
  remapped (see [smolvla.md](smolvla.md) §3.2 for the pattern).

## 5. Results & retrospective

- Full-train and LoRA runs completed (wandb project `toy_project_pi_0`); loss curves
  healthy.
- Pitfalls / lessons:
  1. `compile_model` + PEFT: disable compile for LoRA runs.
  2. Watch reserved (not allocated) memory with `nvitop` — fragmentation after long
     runs can OOM a "fits" config; restart the job between very different recipes.
  3. Weights come from `lerobot/pi0_base` under Gemma Terms of Use — fine for training
     locally, but **do not redistribute weights** in this repo.

## 6. Component breakdown (interview answers)

| Component | In pi0 | My one-line answer |
|---|---|---|
| VLM backbone | PaliGemma ~3B | "Language + vision grounding lives in the VLM; it is finetuned jointly or frozen depending on budget." |
| Action expert | ~300M transformer, flow matching | "A separate lightweight expert predicts the flow-matching velocity field for the action chunk; it attends to VLM KV features like a prefix." |
| Flow matching loss | MSE on velocity v = x₁ − x₀ | "Same objective as SmolVLA — straight-path flows, ~10 integration steps at inference." |
| Memory strategy | grad-checkpoint + bf16 + optional expert-only / LoRA | "Three levers: recompute activations, halve precision, shrink trainable params. Together they fit a 3B VLA in ≤48 GB." |
| LoRA for VLAs | rank-16 on attention projections | "Adapters on the VLM attention layers keep the base frozen — cheap, reversible, and composable per task." |
| pi0 vs SmolVLA | 3B + 300M expert vs 450M total | "Same architectural idea; SmolVLA is the efficient variant that trains on one consumer GPU and deploys inside 8 GB." |
