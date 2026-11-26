# ACT — 用 transformer 做动作分块

> 复现记录（LeRobot 实现）· 在自采 SO-101 数据上训练 · 真机 rollout。
> 论文：Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
> Hardware*（RSS 2023）· 原始代码：tonyzhaozh/act（MIT）· 本次运行用 LeRobot 的
> `policy.type=act`。

## 1. 背景

ACT 用少量低成本遥操作演示，练出精细的操作策略。三个关键设计：

- **动作分块（action chunking）**——每次推理输出一小段*未来动作序列*（一个"块"），
  而不是单步动作，显著稳定长程行为。
- **动作序列上的 CVAE**——变分隐变量捕捉任务级意图，避免解码器对多模态演示做平均。
- **时序集成（temporal ensembling）**——相邻块的重叠预测按指数权重融合，平滑抖动。

**面试官怎么问：**"ACT 的原理是什么？为什么用 CVAE？action chunking 为什么有效？
时序集成怎么实现？"（见 §6 的组件拆解，都是现成答案。）

## 2. 环境与数据

```bash
conda create -y -n lerobot python=3.12 && conda activate lerobot
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[feetech]"
```

数据集：`niuwengang123/lerobot_dataset_put_toy`——任务 *"Put the toy on the plate"*，
约 40 条遥操作 episode，双相机（顶部 + 腕部，640×480 @ 30fps），LeRobotDataset v3.0。
采集流程（标定 → 遥操作 → 录制）见 [../experiments/data_pipeline.md](../experiments/data_pipeline.md)。

## 3. 训练配方

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

| 超参 | 取值 | 说明 |
|---|---|---|
| steps | 20 000 | 这份数据上 loss 早就平了 |
| batch_size | 6 | RTX 3090 上 <10GB 轻松放下 |
| chunk 长度 / KL 权重 / 集成温度 | LeRobot 默认 | 见 `lerobot/policies/act/configuration_act.py` |
| 优化器 / 学习率 | LeRobot 默认 | AdamW |

> GPU-mem：ACT 是小模型（ResNet-18 双视觉骨干 + 约 6 层 transformer），batch 6 峰值
> 远低于 10GB——48GB 预算内毫无压力。精确峰值下次用 `nvitop` 钉一下。

## 4. 推理 / 真机

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

**值得记住的坑（面试也能讲）：** 相机设备号（`/dev/video0` vs `/dev/video4`）在
重启/重插后**不稳定**——录制命令和 rollout 命令里的映射在一次重插后就得对调。
每次运行前用 `v4l2-ctl --list-devices` 确认，且把 `wrist`/`top` 角色映射写在一处。

## 5. 结果与复盘

- 训练收敛平稳；wandb 曲线在 `toy_project_act` 项目里。
- 真机演示：策略能反复完成 *"Put the toy on the plate"*（不同起始位姿，演示视频已录）。
- 待补（v0.1 清单）：按 [real_eval_protocol.md](../experiments/real_eval_protocol.md)
  出正式的 10 次试验成功率表。

踩坑 / 心得：

1. **40 条 episode 的量级，数据质量比模型调参重要得多**——参见 LeRobot 的
   [what makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)
   （复位位姿一致、背景干净、成功轨迹覆盖完整）。
2. 块长度是平滑度与反应速度的折中：太长则机械臂执着于过期计划，太短则行为抖动。
   SO-101 上 LeRobot 默认值就很好。
3. 时序集成几乎零成本、肉眼可见地平滑轨迹——别省。

## 6. 组件拆解（面试答案）

| 组件 | ACT 里的做法 | 我的一句话回答 |
|---|---|---|
| 感知 | 每相机一个 ResNet-18 → 512 维特征，与关节状态拼接 | "多视角 CNN 特征 + 本体感知，投影到 transformer 维度。" |
| 序列模型 | DETR 式 transformer 编码-解码器 | "编码器吃观测；解码器用可学习的动作 query 对编码结果做 cross-attention。" |
| 多模态 | CVAE：动作序列编码器 → 隐变量 z~(μ,σ)，对 N(0,I) 做 KL | "隐变量携带任务级意图，解码器就不会把互相冲突的演示平均掉；推理时取 z=0（先验模式）。" |
| 损失 | L1 重构 + β·KL | "动作块上算 L1，KL 用小权重（LeRobot 默认 10）——隐变量既要有信息又不能飘。" |
| 动作分块 | 预测未来 k 步动作，执行后重规划 | "推理次数更少、行为更平滑，还绕开了单步误差的复利累积。" |
| 时序集成 | 重叠块预测的加权平均 | "w_i ∝ exp(−0.01·i)；融合最近几个块对当前步的预测——便宜且明显更平滑。" |
| 推理循环 | 观测 → 块 → 执行 k 步 → 循环 | "每块一次前向；这个模型尺寸下端到端延迟根本不是问题。" |

**"ACT 和 Diffusion Policy 怎么比？"**——ACT 直接回归动作块（L1）+ CVAE 表达多模态；
Diffusion Policy 用 DDPM/DDIM 对动作块去噪，靠噪声预测目标本身表达多模态。
ACT 更简单、训练更快；DP 的动作分布表达力往往更强。
