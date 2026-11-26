# 架构与设计规则

`so101_vla` 的组织原则：**一条代码路径**覆盖训练、仿真评测、真机部署，且每一行策略代码都能在面试里讲清楚。

## 一条代码路径

所有策略都是 LeRobot `PreTrainedPolicy` 子类（LeRobot 0.6.x API：`PreTrainedConfig` + 前后处理器）。由此保证：

- `lerobot-train --policy.type=<name>` 可训练本仓库任何策略（经 `src/so101_vla/integrations/lerobot_plugin.py` 注册）。
- `lerobot-eval` / `lerobot-rollout` 直接消费同一套 checkpoint——PushT、Meta-World、LIBERO、MuJoCo SO-101 twin、真机共用一个推理契约（`predict_action_chunk` / `select_action`）。
- 新策略即配置 + 模型文件，不写私有训练循环。

## 两个代码区，硬隔离

| 区域 | 内容 | 规则 |
|---|---|---|
| `src/so101_vla/policies/` + `modules/` + `continual/` | 自研代码（从零 mini-VLA、持续学习轨） | 绝不拷贝受限制许可项目的代码（openpi / Gemma ToU、OpenVLA / Llama-2、GR00T / NVIDIA OML、Zeva / OpenMDW）。读论文，写数学。 |
| `src/so101_vla/integrations/` | Apache-2.0 的 LeRobot / SmolVLA 薄封装 | 调用 `lerobot.*`；绝不把其实现代码粘进本仓库。 |
| `docs/repros/` | 经典 VLA **复现展示记录**（配方、结果、拆解） | 只写文档——经典算法不在本仓库做代码重实现。 |

策略类 PR 的评审清单：

1. 实现能否逐模块对应到论文（`docs/repros/<name>.md` 里的对照）？
2. 有没有像从受限仓库拷来的片段？（拒）
3. 依赖里有没有 Apache-2.0 / MIT / BSD 之外的许可？（拒；CI 跑 `pip-licenses`）
4. 有没有硬编码路径、token、密钥？（拒；`tests/test_no_secrets.py`）

## 预算纪律

`configs/train/` 里每个训练配置开头必须有预算注释：

```yaml
# GPU-mem: <峰值>GB @ batch=<b>; fits: 48GB-train / 8GB-infer
```

超出 48GB 训练预算或 8GB 部署预算的配方，评审直接拒。本地开发用单卡 RTX 3090（24GB），
大配方上云 GPU——配置必须可迁移（不写绝对路径）。

## 创新槽：是接口，不是模块

持续学习创新轨统一挂在 `ContinualAdapter` 抽象基类后面
（`src/so101_vla/continual/interface.py`）：

- `lifelong_lora` — 任务级 LoRA 专家累积 + 自动路由（默认）
- `er_replay` — 经验回放基线（必须精心配置：等算力下，调好的 ER 优于联合训练）
- `joint` — 联合训练基线
- `iccl_memory` — 冻结策略的上下文因果记忆（Zeva 式），零梯度对比臂

策略代码不 import 任何具体策略实现，由配置选择。创新轨可插拔，基线因此保持诚实
（同一接口、同一评测）。

## 模块图（从零 mini-VLA，v0.2）

```
观测图像 ─► 视觉编码器（ViT-tiny，多相机 token）
任务文本 ─► 语言编码器（小型 transformer，定长 prefix）
                 │
                 ▼
          融合（action-query token 对视觉+语言做 cross-attention）
                 │
                 ▼
          动作头 ── flow_matching | diffusion | discrete_tokens
                 │
                 ▼
          动作分块（k 步）+ 时序集成 / RTC
```

每个组件对应的面试问题见 `docs/interview/`。

## 真机评测协议

见 `docs/experiments/real_eval_protocol.md`。要点：固定物体位姿、60 秒超时、
每策略 10 次试验、成功=任务终态达成、结果以 JSONL 记录在 `benchmarks/results/`，
汇总进 `benchmarks/RESULTS.md`。
