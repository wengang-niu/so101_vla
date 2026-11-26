# Classic VLA reproductions — showcase records

These are **records of reproductions** (recipes, runs, results, lessons), not
reimplementation code. Each doc is structured for interview use:

1. **Background** — one-paragraph algorithm summary + paper reference + how interviewers ask about it
2. **Environment & data** — setup, dataset (secrets redacted to placeholders)
3. **Training recipe** — copy-pasteable commands, hyperparameter table, GPU-memory budget
4. **Inference / real robot** — rollout commands, camera/port configuration notes
5. **Results & retrospective** — curves, demos, success numbers, pitfalls worth talking about
6. **Component breakdown** — encoder / action head / loss / inference loop in my own words (interview answers)

| Doc | Algorithm | Status |
|---|---|---|
| [act.md](act.md) | ACT — action chunking with transformers (CVAE + temporal ensembling) | trained + real-robot rollout |
| [smolvla.md](smolvla.md) | SmolVLA — 0.5B VLA with flow-matching action expert | full-train + finetune + real-robot rollout |
| [pi0.md](pi0.md) | pi0 — flow-matching action expert on a VLM backbone | full-train + LoRA finetune (inference TODO) |
| `pi05_lora_libero.md` | pi0.5-LoRA on LIBERO (parity band 96.9–97.5%) | planned (v0.5+) |
