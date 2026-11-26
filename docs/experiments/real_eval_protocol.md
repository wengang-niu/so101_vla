# 真机评测协议（SO-101）

本仓库所有策略共用一套协议，保证成功率数字在复现之间、研究变体之间可比。
默认任务：**"Put the toy on the plate"**。

## 协议

1. **准备：** 固定并标记物体起始位姿（3 个标准位姿 + 2 个标记区内随机位姿）；
   光照/背景与数据采集一致；机械臂已回零并标定。
2. **试验次数：** 每策略每 checkpoint 10 次。顺序：标准/随机位姿交替（不连续同起始）。
3. **超时：** 每次 60 秒。
4. **成功判据：** episode 结束时玩具完全在盘子上，录像确认（物体完全在盘边界内；
   盘子未被推离原位）。
5. **失败分类**（逐次记录）：`grasp_fail` / `transport_fail` / `place_fail` /
   `timeout` / `hardware`（硬件故障剔除重跑）。
6. **录像：** 每次试验都录；归档在 `benchmarks/results/videos/`（不入库——外链或
   外挂附件）。

## 日志格式（JSONL）

每个"策略-checkpoint-日期"一个文件：
`benchmarks/results/real_<task>_<policy>_<YYYYMMDD>.jsonl`

```json
{
  "trial_id": 1,
  "policy": "smolvla",
  "checkpoint": "smolvla_040000/040000/pretrained_model",
  "task": "Put the toy on the plate",
  "start_pose": "canonical_1",
  "success": true,
  "failure_mode": null,
  "duration_s": 23.4,
  "video": "benchmarks/results/videos/20261126_smolvla_trial01.mp4",
  "notes": ""
}
```

## 汇报

汇总进 `benchmarks/RESULTS.md`：

| 策略 | Checkpoint | 成功率 | 失败模式 | 日期 |
|---|---|---|---|---|
| （每次运行后填） | | x/10 | | |

对比规则：两个策略正面对比时用同一批位姿、最好同一天；做不到就在表旁注明差异。
绝不把一次侥幸成功当结果报——最少 10 次试验，负样本照实记录。
