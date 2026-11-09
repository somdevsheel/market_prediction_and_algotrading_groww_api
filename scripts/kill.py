"""Kill switch CLI.  python scripts/kill.py --on "reason"   |   python scripts/kill.py --off   |   --status"""
import _bootstrap  # noqa: F401

import argparse

from core.config import load_config, resolve_path
from risk.kill_switch import KillSwitch

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--on", metavar="REASON")
g.add_argument("--off", action="store_true")
g.add_argument("--status", action="store_true")
a = ap.parse_args()
k = KillSwitch(resolve_path(load_config(), "kill_switch_file"))
if a.on:
    k.activate(a.on, "cli")
    print("KILL SWITCH ON — the running worker will exit any position and block entries.")
elif a.off:
    k.deactivate()
    print("kill switch off")
print("status:", k.info() or "inactive")
