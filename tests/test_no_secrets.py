"""扫描所有被 git 跟踪的文本文件，拒绝任何密钥形状的字符串。

在 CI 与 pre-commit 中自动运行，是硬门禁：命中一条就让构建失败。规则刻意写得通用
（HF token、W&B key、OpenAI 风格 key、AWS access key、私钥块）。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each pattern is built so the literal source of this file cannot self-match.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("huggingface-token", re.compile(r"hf" + r"_[A-Za-z0-9]{30,}")),
    ("wandb-key", re.compile(r"wandb" + r"_[0-9a-f]{32,}")),
    ("openai-style-key", re.compile(r"\bsk" + r"-[A-Za-z0-9]{20,}")),
    ("aws-access-key", re.compile(r"\bAKIA" + r"[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"-----BEGIN" + r" (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "outputs", "data", "wandb", "checkpoints"}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".pdf", ".pt", ".pth", ".ckpt",
    ".safetensors", ".onnx", ".bin", ".pkl", ".zip", ".gz",
}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [REPO_ROOT / f for f in out.split("\0") if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback (e.g. running outside git): walk the tree.
        files = []
        for p in REPO_ROOT.rglob("*"):
            if p.is_file() and not (set(p.parts) & SKIP_DIRS) and p.suffix not in BINARY_SUFFIXES:
                files.append(p)
        return files


def test_no_secrets_in_tracked_files() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if path.suffix in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no} matches {name}")
    assert not offenders, "Potential secrets found:\n" + "\n".join(offenders)
