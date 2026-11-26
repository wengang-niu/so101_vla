# SO-101 data pipeline record

Hardware → calibration → teleop → collection → dataset format → visualization →
upload → monitoring. All credentials are redacted to placeholders; use
`hf auth login` / `wandb login` interactively.

## Hardware & software

- SO-101 leader arm (teleop) + follower arm (robot), Feetech STS3215 servos.
  Leader: `/dev/ttyACM0` (id `g_leader_arm`); follower: `/dev/ttyACM1` (id `g_follower_arm`).
- Two USB cameras: `top` + `wrist`, 640×480 @ 30 fps, MJPG fourcc.
- Environment: `conda create -y -n lerobot python=3.12`, then
  `git clone https://github.com/huggingface/lerobot.git && pip install -e ".[feetech]"`.
- Sanity check: `python -c "import torch, lerobot; print(torch.cuda.is_available(), lerobot.__version__)"`.

## Calibration

```bash
ls /dev/ttyACM*

lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 --teleop.id=g_leader_arm

lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 --robot.id=g_follower_arm
```

Recalibrate after any mechanical change (homing shifts = silent policy degradation).

## Teleoperation

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=g_follower_arm \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=g_leader_arm
```

## Recording

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

Collection quality rules (from LeRobot's
[dataset guide](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)):
consistent reset poses, diverse-but-representative object placements, clean successful
demos, consistent lighting/background.

**Time alignment:** cameras run at 30 Hz, joint state at 100 Hz; LeRobotDataset
samples carry timestamps so policies can resample to their control rate. Keep camera
`fps` identical across recording and rollout.

**Camera-index gotcha:** `/dev/video*` indices shuffle across reboots/replugs. Verify
with `v4l2-ctl --list-devices` before every session; the `wrist`/`top` role mapping is
what matters, not the index.

## Dataset format (LeRobotDataset v3.0)

- `data/` — one parquet row per frame (action, `observation.state`, timestamps, indices)
- `videos/` — one subdirectory per camera; frames encoded into mp4 (av1/yuv420p/30fps)
- `meta/` — `info.json` (schema/id card), `stats.json` (normalization statistics),
  `tasks.parquet` (task strings), `episodes/` (per-episode metadata)

## Visualization

- HF Space: https://huggingface.co/spaces/lerobot/visualize_dataset
- Alternative viewer: http://io-ai.tech/lerobot
- TODO: walkthrough article supplement (noted for a later blog post).

## Upload / download (Hugging Face)

```bash
pip install huggingface_hub
hf auth login                 # interactive; NEVER hardcode tokens
hf upload <user>/lerobot_dataset_put_toy ./data/ --repo-type=dataset
hf download --repo-type dataset <user>/lerobot_dataset_put_toy --local-dir ./data
```

## Monitoring

```bash
pip install wandb && wandb login    # interactive; NEVER hardcode API keys
pip install nvitop                   # GPU memory/utilization watch
```

Rules: credentials live in `~/.netrc` / env vars / `wandb login`, never in scripts or
docs. This repo's CI fails on credential-shaped strings (`tests/test_no_secrets.py`).
