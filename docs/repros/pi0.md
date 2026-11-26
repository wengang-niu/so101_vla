# pi0 — VLM 基座 + flow-matching 动作专家

> 复现记录（LeRobot `policy.type=pi0`）· 从 `lerobot/pi0_base` 全量训练 + LoRA 微调
> （自采 SO-101 数据）· 推理待补。
> 论文：Black et al., *π0: A Vision-Language-Action Flow Model for General Robot
> Control*（Physical Intelligence，2024）· 权重经 LeRobot 获取，受 Gemma Terms of Use
> 约束（权重**不在**本仓库转分发）。

## 1. 背景

pi0 把 PaliGemma 级 VLM（≈3B）与轻量**动作专家**（≈300M）配对，动作专家用
**flow matching** 生成动作块，训练形式是 prefix-LM——专家关注 VLM 特征。它是
"大 VLA"档位里单机预算还能微调的最强开源配方——靠显存技巧。pi0.5（同架构家族的
后续工作，本路线后面会记录）在长程/语言落地能力上有提升。

**面试官怎么问：**"pi0 和 SmolVLA 区别？action expert 为什么单独设计？LoRA 微调 VLA
怎么配？全量微调 48GB 怎么塞进去？"（见 §6。）

## 2. 环境与数据

```bash
pip install -e ".[pi]"        # LeRobot 的 pi0 依赖
pip install 'lerobot[dataset]'
pip install 'lerobot[training]'
pip install 'lerobot[peft]'   # 仅 LoRA 微调需要
```

数据集与 [act.md](act.md) 相同。全量训练跑在云 GPU 节点（48GB 级）；LoRA 微调在
单张 RTX 3090 上即可。

## 3. 训练配方

### 3.1 全量训练（带显存技巧）

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

### 3.2 LoRA 微调

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

| 超参 | 取值 | 为什么重要 |
|---|---|---|
| `gradient_checkpointing=true` | 两者 | 反向时重算激活——单项最大的显存杠杆（约省 2–3 倍） |
| `dtype=bfloat16` | 两者 | 比 fp32 省一半优化器/激活显存；用 bf16 而非 fp16 是为了避免 VLM 溢出 |
| `compile_model=true` | 仅全量 | `torch.compile` 内核；LoRA 时关掉——compile 与 PEFT 钩子相互干扰 |
| `freeze_vision_encoder=false` | 全量 | 全量时连视觉塔一起更新；冻住它是更轻的选择 |
| `train_expert_only=false` | 全量 | 训练 VLM+专家；`true` = 只训专家，最省 |
| `peft.r=16 / lora_alpha=16` | LoRA | 注意力投影上的 rank-16 适配器，可训练参数约 1% |
| steps | 50 000 / 3 000 | LoRA 收敛所需步数少一个量级 |

> GPU-mem：全量微调 batch 8 + 上述技巧可放进 48GB 节点（精确峰值待从运行日志钉）；
> LoRA 微调峰值远低于 24GB，RTX 3090 可跑。这套配方就是本仓库"3B VLA 怎么塞进
> ≤48GB"的面试答案。

## 4. 推理 / 真机 — **待补（已记录的缺口）**

pi0 的 SO-101 rollout **尚未记录**（笔记中的已知缺口）。补上后本节会有
`lerobot-rollout` 命令与结果。先记住几个约束：

- pi0 bf16 推理 **>8GB 显存**——开箱进不了 8GB 真机预算。可选方案：int8 量化、
  ONNX/TensorRT 导出、或 LeRobot 的远程推理拆分（策略服务器在云 GPU，臂侧只跑客户端）。
  三条路都在部署轨计划里。
- 若训练时做了相机名重映射，rollout 必须沿用同样的约定（模式见
  [smolvla.md](smolvla.md) §3.2）。

## 5. 结果与复盘

- 全量与 LoRA 运行均完成（wandb 项目 `toy_project_pi_0`），loss 曲线健康。
- 踩坑 / 心得：
  1. `compile_model` 与 PEFT：LoRA 运行要关 compile。
  2. 用 `nvitop` 盯 reserved（不是 allocated）显存——长任务后的碎片化可能让"放得下"
     的配置 OOM；切换差异大的配方前重启任务。
  3. `lerobot/pi0_base` 权重受 Gemma Terms of Use 约束——本地训练没问题，但**绝不
     在本仓库转分发权重**。

## 6. 组件拆解（面试答案）

| 组件 | pi0 里的做法 | 我的一句话回答 |
|---|---|---|
| VLM 基座 | PaliGemma ~3B | "语言+视觉落地在 VLM 里；按预算决定联合微调还是冻结。" |
| 动作专家 | ~300M transformer，flow matching | "独立轻量专家预测动作块的流匹配速度场；像 prefix 一样关注 VLM 的 KV 特征。" |
| flow matching 损失 | 速度 v=x₁−x₀ 的 MSE | "与 SmolVLA 同一目标——直线路径流，推理约 10 步积分。" |
| 显存策略 | 梯度检查点 + bf16 + 可选专家-only / LoRA | "三个杠杆：重算激活、减半精度、缩小可训练参数。合起来让 3B VLA 装进 ≤48GB。" |
| VLA 上的 LoRA | 注意力投影 rank-16 | "在 VLM 注意力层挂适配器，基座冻结——便宜、可逆、按任务可组合。" |
| pi0 vs SmolVLA | 3B+300M 专家 vs 总计 450M | "同一架构思想；SmolVLA 是高效变体，单张消费级卡可训、8GB 内可部署。" |
