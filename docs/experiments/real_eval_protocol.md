# Real-robot evaluation protocol (SO-101)

One protocol for every policy in this repo, so success numbers are comparable across
reproductions and research variants. Default task: **"Put the toy on the plate"**.

## Protocol

1. **Setup:** fixed, marked object start poses (record 3 canonical poses + 2 random
   within the marked zone); same lighting/background as data collection; arm homed and
   calibrated.
2. **Trials:** 10 per policy per checkpoint. Trial order: interleave canonical/random
   poses (no back-to-back identical starts).
3. **Timeout:** 60 s per trial.
4. **Success criterion:** toy fully on the plate at the end of the episode, verified on
   video (object completely within the plate boundary; plate not displaced).
5. **Failure taxonomy** (record per trial): `grasp_fail` / `transport_fail` /
   `place_fail` / `timeout` / `hardware` (hardware failures are excluded and re-run).
6. **Video:** record every trial; archive under `benchmarks/results/videos/` (not
   committed — link or attach externally).

## Logging format (JSONL)

One file per policy-checkpoint-date:
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

## Reporting

Summarize in `benchmarks/RESULTS.md`:

| Policy | Checkpoint | Success | Failure modes | Date |
|---|---|---|---|---|
| (to be filled per run) | | x/10 | | |

Comparison rules: same poses, same day when comparing two policies head-to-head; if
not possible, note the discrepancy next to the table. Never report a single lucky
episode as a result — 10 trials minimum, negatives included.
