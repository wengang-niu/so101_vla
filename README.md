# so101_vla

**经典 VLA（视觉-语言-动作）算法复现实录（面试展示用）+ 持续学习 VLA 自研路线**——训练 ≤48GB 显存，部署在自攒 SO-101 机械臂（8GB 显卡）上。

本仓库有两个目标：

1. **经典 VLA 算法复现，记录成文档展示。** ACT、SmolVLA、pi0 的完整训练/推理配方、SO-101 真机实操、结果与踩坑复盘，每篇附"组件拆解"——按面试问答的方式把算法讲透。见 [`docs/repros/`](docs/repros/)。
2. **自研/改进方法（面试 + 研究）。** 创新方向为**持续学习（continual / lifelong learning）**：任务级 LoRA 专家累积 + 新颖度门控回放，在 MuJoCo + 真机 SO-101 顺序任务流上建 benchmark。另有一个从零实现的 mini-VLA 作为学习载体，每个组件（编码器 → 融合 → 动作头 → 推理循环）都为"能讲清楚"而写。目前自研部分先出计划，见 [`docs/plans/`](docs/plans/)。

一切基于 [LeRobot](https://github.com/huggingface/lerobot) 技术栈，**同一条代码路径**覆盖 `lerobot-train` → `lerobot-eval`（PushT / Meta-World / LIBERO）→ `lerobot-rollout` 真机部署。

## 现状

早期开发中。硬约束：

| 约束 | 数值 |
|---|---|
| 训练预算 | ≤48GB 显存（本地 RTX 3090 + 云 GPU，配方可迁移） |
| 真机推理 | ≤8GB 显存（x86 主机 + RTX 4060 级显卡） |
| 机械臂 | 自攒 SO-101（LeRobot 生态），腕部 + 顶部双相机 |
| 仿真 | MuJoCo（LeRobot SO-101 twin，`lerobot-eval` 跑 LIBERO / Meta-World） |
| License | Apache-2.0（代码）。受限制的第三方权重一律不转分发，见下文 |

## 复现展示

| 算法 | 记录内容 | 文档 |
|---|---|---|
| **ACT** | 训练 + SO-101 真机（"Put the toy on the plate"），CVAE/动作分块拆解 | [docs/repros/act.md](docs/repros/act.md) |
| **SmolVLA** | 从零训练 + 基座微调配方、相机名映射、fp16 部署数字 | [docs/repros/smolvla.md](docs/repros/smolvla.md) |
| **pi0** | 从 `lerobot/pi0_base` 全量训练 + LoRA 微调配方 | [docs/repros/pi0.md](docs/repros/pi0.md) |

数据采集与硬件搭建（标定、遥操作、LeRobotDataset v3.0、时间对齐）记录在
[docs/experiments/data_pipeline.md](docs/experiments/data_pipeline.md)。
真机评测协议在 [docs/experiments/real_eval_protocol.md](docs/experiments/real_eval_protocol.md)。

## 路线图

| 版本 | 主题 |
|---|---|
| **v0.1** | 脚手架 + 经典复现展示文档 + 数据管线记录 + 真机评测协议 |
| **v0.2** | 从零 mini-VLA（flow-matching 动作头）+ 面试文档系列 |
| **v0.3** | 持续学习自研轨：LoRA 专家累积 + 门控回放 + 真机持续学习 benchmark |
| **v0.4** | 部署强化：ONNX fp16/int8、延迟表、Zenoh 远程推理 |
| v0.5+ | pi0.5-LoRA LIBERO 训练记录、博客系列、可选基线行 |

## 目录结构

```
docs/
  repros/            # ★ 经典 VLA 复现记录（配方、结果、面试拆解）
  experiments/       # 数据管线、真机评测协议、实验记录
  interview/         # 面试问答文档（旗舰：从零构建一个 VLA）
  plans/             # 自研计划（持续学习路线）
  architecture.md    # 设计规则
src/so101_vla/
  policies/          # 从零 mini-VLA（LeRobot PreTrainedPolicy 子类）
  modules/           # 积木：视觉、语言、融合、动作头、时序
  continual/         # 创新轨：ContinualAdapter 接口 + 策略
  integrations/      # LeRobot 薄封装（不拷代码）
  deployment/        # ONNX 导出、量化、延迟基准
configs/train/       # 每个配方一个 YAML，头部带 # GPU-mem 预算注释
scripts/             # 评测 / rollout 脚本
benchmarks/          # 结果汇总（JSONL + markdown，永不提交权重）
tests/               # 单元测试 + 密钥扫描
```

## 快速开始

```bash
# 1. 环境（Python >= 3.10）
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 冒烟测试（无需 GPU）
pytest

# 3. 阅读复现记录
open docs/repros/act.md

# 4. 真机 rollout 并记录成功率（训练好策略之后）
bash scripts/eval_real_so101.sh <policy>
```

本机训练硬件：1 张 RTX 3090（24GB）本地 + 云 GPU 跑大配方。

## License 与第三方权重

本仓库代码为 **Apache-2.0**。不 vendor 任何受限制许可的代码或权重
（openpi / Gemma ToU、OpenVLA / Llama-2 社区许可、GR00T / NVIDIA Open Model License、
Zeva / OpenMDW）。使用此类模型时（如 pi0 配方经由 LeRobot），按其自身条款运行，
权重绝不在此仓库转分发。

## 安全

**绝不提交凭据。** CI 会扫描密钥（`tests/test_no_secrets.py` + gitleaks pre-commit）。
Hugging Face / W&B 一律用 `hf auth login`、`wandb login` 或环境变量——文档中的命令
全部使用占位符。
