# SmolVLA — 小参数量 VLA + flow-matching 动作专家

> 复现记录（LeRobot）· 在自采 SO-101 数据上从零训练 + 基座微调 · 真机 rollout ·
> fp16 部署在 8GB 预算内。
> 论文：SmolVLA（Hugging Face LeRobot 团队，2025）· 权重：`lerobot/smolvla_base`
> （代码与权重均为 Apache-2.0）· 经 `policy.type=smolvla` 运行。

## 1. 背景

SmolVLA 是约 450M 参数的 VLA，为高效训练与机上推理而生：

- **SmolVLM2** 视觉-语言基座（0.5B 级）——小到单卡可微调，强到能在杂乱桌面场景里
  把语言落到空间。
- **flow-matching 动作专家**（pi0 式）：一个小 transformer 专家预测动作块"流"的
  *速度*；推理时积分约 10 步出动作。
- **动作分块** + 流式推理，为实时控制设计。

**面试官怎么问：**"SmolVLA 和 pi0 什么关系？flow matching 怎么推导？动作头为什么叫
action expert？微调时相机名对不上怎么办？"（见 §6。）

## 2. 环境与数据

与 [act.md](act.md) 相同（LeRobot + SO-101，数据集
`niuwengang123/lerobot_dataset_put_toy`，双相机）。SmolVLA 不引入新依赖——它内置在
LeRobot 里。

## 3. 训练配方

### 3.1 从零训练

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

### 3.2 从 `lerobot/smolvla_base` 微调

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

| 超参 | 取值 | 说明 |
|---|---|---|
| steps | 40 000 | 微调比从零快得多；更早的 checkpoint 已经能用 |
| batch_size | 8 | 这个尺寸约 <24GB（RTX 3090 能跑） |
| `--rename_map` | top→camera1、wrist→camera2 | 预训练 checkpoint 期望 `camera1/camera2` 命名 |
| `--policy.empty_cameras=1` | 1 | 补一个缺失相机槽位，保证 checkpoint 的相机配置对得上 |

**rename_map 为什么存在（高频"代码细节"考点）：** 基座 checkpoint 预训练时的相机槽位
模式是固定的（`observation.images.camera1`、`camera2`），而我们的数据集叫
`top`/`wrist`。`--rename_map` 把数据集键名重映射到 checkpoint 期望的模式；
`--policy.empty_cameras=1` 在相机数量不一致时保持配置一致。忘了这两个参数，是
SmolVLA 微调"悄悄变差"的最常见原因。

## 4. 推理 / 真机

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

rollout 时必须带**同一个** `--rename_map`——checkpoint 里存的策略配置指的是
`camera1/camera2`。

## 5. 结果与复盘

- 两种配方都训到收敛（wandb 项目 `toy_project_smolvla`）；微调远早于从零可用。
- 真机演示完成 *"Put the toy on the plate"*（演示视频已录）。
- **部署：** fp16 推理约 1–2GB 显存——RTX 4060 级 8GB 卡上非常宽裕。这就是本仓库
  把 SmolVLA 当部署主力、而 pi0 级模型只做训练侧的原因。
- 待补（v0.1 清单）：正式 10 次试验成功率表
  （[real_eval_protocol.md](../experiments/real_eval_protocol.md)）。

踩坑 / 心得：

1. 相机槽位不匹配（见 §3.2）——微调第一大坑。
2. 约 40 条 episode 的量级下，从 `smolvla_base` 微调是明显正确的选择；从零训练只
   适合研究架构本身。
3. rollout 的 `task` 字符串要与数据集 `single_task` 措辞一致——语言通道对措辞敏感。

## 6. 组件拆解（面试答案）

| 组件 | SmolVLA 里的做法 | 我的一句话回答 |
|---|---|---|
| VLM 基座 | SmolVLM2 ~450M（图像 token + 文本） | "紧凑 VLM 提供语言落地与场景理解；这里是微调而不是预训练。" |
| 动作专家 | 小 transformer，**flow matching** 训练 | "pi0 式专家：不用 DDPM 去噪，而是回归直线流 x_t=(1−t)x₀+t·x₁ 的速度场 v=x₁−x₀；推理积分约 10 个 Euler 步。" |
| 分块 / 执行 | 预测动作块，流入控制器 | "每块一次流积分；便宜到可以闭环重规划。" |
| 损失 | 速度的 MSE | "E[‖v̂(x_t,t,obs)−(x₁−x₀)‖²]；比 diffusion 简单——不用调噪声调度。" |
| 配置层集成 | `rename_map`、`empty_cameras` | "预训练 checkpoint 有自己的相机槽位模式；把数据集键名映射过去，否则性能悄悄掉。" |
| 部署 | fp16 1–2GB | "8GB 卡装下它加相机流绰绰有余；int8/ONNX 量化是下一步，不是必需。" |

**"flow matching 和 diffusion 什么区别？"**——两者都是学噪声到数据的映射；diffusion
在随机调度上学噪声预测目标，flow matching 沿直线路径回归确定性速度场。flow matching
通常需要更少推理步数、活动部件更少——所以 pi0/SmolVLA 系动作专家用它。
