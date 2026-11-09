"""Path setup: this experiment lives inside the main project but never writes to it.

* EXP_ROOT  : this folder (all outputs, configs and its own data store live here)
* MAIN_ROOT : the main project (imported read-only for its Groww provider, data-lake class and calendar)
"""
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
MAIN_ROOT = EXP_ROOT.parent.parent
for p in (str(MAIN_ROOT), str(EXP_ROOT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
STORE = EXP_ROOT / "data" / "storage"
RESULTS = EXP_ROOT / "results"
