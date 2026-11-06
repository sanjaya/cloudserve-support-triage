"""Check that no API key, token or private key is in the repository (NFR-04; backlog item B-13).

    python scripts/check_no_secrets.py             # files git tracks now
    python scripts/check_no_secrets.py --history   # also every line ever added in the git history

The Project Brief: "No credential, key or token may appear anywhere in your committed source
code." CI runs this on every push, with the full history. It looks for the shapes of real keys
(OpenRouter, OpenAI, Anthropic, Groq, Hugging Face, GitHub, AWS, private-key files) and for a
.env file being tracked. Placeholders such as "your_key_here" are fine.

Exit code 0 = nothing found; 1 = something found (the line is shown with the secret masked).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = [
    ("OpenRouter key", re.compile(r"sk-or-v1-[A-Za-z0-9]{32,}")),
    ("Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{30,}")),
    ("OpenAI key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("Groq key", re.compile(r"\bgsk_[A-Za-z0-9]{40,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("key assigned in code or config", re.compile(
        r"(?i)\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\b\s*[:=]\s*['\"]?(?!your_|<|\$\{|\$|none\b|null\b|test\b|x+\b)"
        r"[A-Za-z0-9_\-]{24,}")),
]


def mask(text: str) -> str:
    return text[:6] + "…" + f"({len(text)} chars)"


def scan_text(text: str) -> list[tuple[str, str]]:
    found = []
    for name, pattern in PATTERNS:
        for m in pattern.finditer(text):
            found.append((name, mask(m.group(0))))
    return found


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def scan_tracked() -> list[str]:
    problems = []
    files = tracked_files()
    for f in files:
        name = Path(f).name
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            problems.append(f"{f}: a .env file is tracked by git - remove it (git rm --cached {f})")
        path = ROOT / f
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue                     # binary files such as the trained model
        for n, line in enumerate(text.splitlines(), 1):
            for kind, shown in scan_text(line):
                problems.append(f"{f}:{n}: {kind} {shown}")
    return problems


def scan_history() -> list[str]:
    """Every line added in every commit (git log -p), so a key that was committed and later
    deleted is still found."""
    log = subprocess.run(["git", "log", "--all", "-p", "--no-color", "--unified=0", "--format=commit %h"],
                         cwd=ROOT, capture_output=True, text=True, errors="replace", check=True).stdout
    problems, commit, current = [], "?", "?"
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++"):
            for kind, shown in scan_text(line[1:]):
                problems.append(f"history {commit} {current}: {kind} {shown}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", action="store_true", help="also scan every commit in the git history")
    args = ap.parse_args(argv)
    problems = scan_tracked()
    if args.history:
        problems += scan_history()
    files = len(tracked_files())
    if problems:
        print(f"Possible secrets found ({len(problems)}):")
        for p in problems:
            print("  " + p)
        return 1
    print(f"No keys found in {files} tracked files" + (" or in the git history." if args.history else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
