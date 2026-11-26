# SO-101 数据管线记录

硬件 → 标定 → 遥操作 → 采集 → 数据格式 → 可视化 → 上传 → 监控。所有凭据已脱敏为
占位符；请交互式执行 `hf auth login` / `wandb login`。

## 硬件与软件

- SO-101 leader 臂（遥操作）+ follower 臂（机器人），Feetech STS3215 舵机。
  leader：`/dev/ttyACM0`（id `g_leader_arm`）；follower：`/dev/ttyACM1`（id `g_follower_arm`）。
- 两路 USB 相机：`top` + `wrist`，640×480 @ 30fps，MJPG fourcc。
- 环境：`conda create -y -n lerobot python=3.12`，然后
  `git clone https://github.com/huggingface/lerobot.git && pip install -e ".[feetech]"`。
- 自检：`python -c "import torch, lerobot; print(torch.cuda.is_available(), lerobot.__version__)"`。

## 标定

```bash
ls /dev/ttyACM*

lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 --teleop.id=g_leader_arm

lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 --robot.id=g_follower_arm
```

任何机械结构变动后重新标定（回零偏移 = 策略性能悄悄下降）。

## 遥操作

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=g_follower_arm \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=g_leader_arm
```

## 录制

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=g_follower_arm \
  --robot.cameras='{"wrist":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"fourcc":"MJPG"},"top":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=g_leader_arm \
  --dataset.repo_id=<user>/lerobot_dataset_put_toy \
  --dataset.root=$DATA_ROOT/lerobot_dataset_put_toy \
  --dataset.num_episodes=40 \
  --dataset.single_task="Put the toy on the plate" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

采集质量守则（来自 LeRobot 的
[数据集指南](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)）：
复位位姿一致、物体摆放有代表性地多样、干净的成功演示、光照/背景一致。

**时间对齐：** 相机 30Hz、关节状态 100Hz；LeRobotDataset 样本自带时间戳，策略可按
自己的控制频率重采样。录制与 rollout 的相机 `fps` 必须一致。

**相机设备号的坑：** `/dev/video*` 序号在重启/重插后会变。每次开跑前用
`v4l2-ctl --list-devices` 确认；关键是 `wrist`/`top` 的角色映射，不是序号。

## 数据格式（LeRobotDataset v3.0）

- `data/` — 每帧一行 parquet（action、`observation.state`、时间戳、各类索引）
- `videos/` — 每相机一个子目录；帧编码进 mp4（av1/yuv420p/30fps）
- `meta/` — `info.json`（schema/身份信息）、`stats.json`（归一化统计）、
  `tasks.parquet`（任务文本）、`episodes/`（每集元数据）

## 可视化

- HF Space：https://huggingface.co/spaces/lerobot/visualize_dataset
- 备用查看器：http://io-ai.tech/lerobot
- 待补：配一篇图解补充文章（留给后面的博客）。

## 上传 / 下载（Hugging Face）

```bash
pip install huggingface_hub
hf auth login                 # 交互式；绝不硬编码 token
hf upload <user>/lerobot_dataset_put_toy ./data/ --repo-type=dataset
hf download --repo-type dataset <user>/lerobot_dataset_put_toy --local-dir ./data
```

## 监控

```bash
pip install wandb && wandb login    # 交互式；绝不硬编码 API key
pip install nvitop                   # 盯显存/利用率
```

守则：凭据放 `~/.netrc` / 环境变量 / `wandb login`，绝不进脚本和文档。本仓库 CI 会
拒绝任何密钥形状的字符串（`tests/test_no_secrets.py`）。
