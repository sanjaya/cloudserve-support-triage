"""The kill switch (backlog item B-21; Governance Framework section 5).

Stops all automatic answers immediately, without a deployment or a restart. While it is on,
every ticket still goes through classification and search (so the person gets a full hand-off
note), but nothing is sent to a customer: decide sends every ticket to a person, and the reply
step checks again just before sending, so a ticket that was already being answered when the
switch was turned on is also stopped.

Two ways to turn it on (either is enough):
- the file storage/KILL_SWITCH exists (checked for every ticket, so it takes effect on the next
  ticket of a run that is already going);   python -m src.kill_switch on --by "Name" --reason "..."
- the setting KILL_SWITCH=1 in .env or the environment (takes effect when the system starts).

    python -m src.kill_switch status
    python -m src.kill_switch on --by "Marcus Adeyemi" --reason "wrong answers reported"
    python -m src.kill_switch off --by "Marcus Adeyemi"

Every change is appended to storage/kill_switch_log.txt (who, when, why).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from src import config


def _file() -> Path:
    return Path(os.getenv("KILL_SWITCH_FILE") or (config.STORAGE_DIR / "KILL_SWITCH"))


def _log() -> Path:
    return _file().with_name("kill_switch_log.txt")


def is_on() -> bool:
    """True if automatic answers are switched off. Checked for every ticket."""
    if os.getenv("KILL_SWITCH", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return _file().exists()


def why() -> str:
    if os.getenv("KILL_SWITCH", "").strip().lower() in ("1", "true", "yes", "on"):
        return "KILL_SWITCH setting"
    try:
        return _file().read_text(encoding="utf-8").strip() or "kill switch file present"
    except OSError:
        return "kill switch off"


def set_switch(on: bool, by: str, reason: str = "") -> str:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{stamp} {'ON' if on else 'OFF'} by {by}" + (f": {reason}" if reason else "")
    if on:
        f.write_text(line + "\n", encoding="utf-8")
    elif f.exists():
        f.unlink()
    with open(_log(), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Switch automatic answers off (on) or back on (off).")
    ap.add_argument("action", choices=["on", "off", "status"])
    ap.add_argument("--by", default=os.getenv("USER", "unknown"), help="who is doing this")
    ap.add_argument("--reason", default="")
    args = ap.parse_args(argv)
    if args.action == "status":
        print(f"Kill switch is {'ON – no automatic answers' if is_on() else 'off – automatic answers allowed'}"
              f"{' (' + why() + ')' if is_on() else ''}")
        return
    print(set_switch(args.action == "on", args.by, args.reason))
    if args.action == "off" and is_on():
        print("Note: still ON because KILL_SWITCH is set in the environment or .env.")


if __name__ == "__main__":
    main()
