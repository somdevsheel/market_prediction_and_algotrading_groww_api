"""Download NSE "Integrated Filing - Financials" (quarterly results for periods from Mar 2025 on) - RUN IT YOURSELF.

The older endpoint (api/corporates-financial-results, used by ~/Project/Quantera) stops at Dec 2024 quarters,
because SEBI moved results into Integrated Filing in 2025. This script fetches only the newer filings, for the
NSE stocks in our model universe, and saves the raw files. Parsing is a separate step.

Polite by design: one request at a time, PAUSE seconds apart, and it STOPS (no retries, no workarounds) the first
time NSE answers 401 / 403 / 429. It is restartable: files already saved are skipped, so just run it again later.

Saves:
  data_lake/bronze/india_fundamentals/integrated/list/<SYMBOL>.json     the filing list NSE returns
  data_lake/bronze/india_fundamentals/integrated/xbrl/<SYMBOL>/<file>.xml
  data_lake/bronze/india_fundamentals/integrated/log.csv

    python3 -m ml_lab.data.nse_integrated_filings              # all ~1,200 universe stocks (~4 h, ~12 GB)
    python3 -m ml_lab.data.nse_integrated_filings --limit 5    # try a few first
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from core.config import load_config, resolve_path

LAKE = resolve_path(load_config(), "data_lake")
BRONZE = LAKE / "bronze" / "india_fundamentals" / "integrated"
PAGE = "https://www.nseindia.com/companies-listing/corporate-filings-integrated-filing"
API = "https://www.nseindia.com/api/integrated-filing-results"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Referer": PAGE}
PAUSE = 1.5


class Blocked(Exception):
    pass


def universe() -> list[str]:
    """NSE symbols in the latest ML signals (the stocks the model actually ranks), most liquid first."""
    runs = sorted((Path(__file__).resolve().parents[1] / "results").glob("bhs_*/signals_history/*.csv"))
    s = pd.read_csv(runs[-1]).sort_values("tv", ascending=False)
    ex = pd.read_parquet(LAKE / "silver" / "yahoo_stocks" / "all_daily.parquet", columns=["sym", "primary_exchange"]).drop_duplicates("sym")
    nse = set(ex.sym[ex.primary_exchange == "NSE"])
    return [x for x in s.sym if x in nse]


def get(sess: requests.Session, url: str, **kw) -> requests.Response:
    time.sleep(PAUSE)
    r = sess.get(url, timeout=(10, 60), **kw)
    if r.status_code in (401, 403, 429):
        raise Blocked(f"NSE answered HTTP {r.status_code} for {url}")
    return r


_OLD_BASIS: dict | None = None


def old_basis() -> dict:
    """Basis each company used in the pre-2025 panel (so year-on-year growth compares like with like)."""
    global _OLD_BASIS
    if _OLD_BASIS is None:
        p = LAKE / "gold" / "india_fundamentals" / "fund_panel.parquet"
        _OLD_BASIS = (pd.read_parquet(p, columns=["symbol", "consolidated"]).groupby("symbol").consolidated.last()
                      .map({"Consolidated": "Consolidated", "Non-Consolidated": "Standalone"}).to_dict()) if p.exists() else {}
    return _OLD_BASIS


def pick_one_per_quarter(rows: list) -> list[dict]:
    """One XBRL per quarter end: the same basis the company had in the pre-2025 data (else Consolidated if it
    files consolidated), the earliest 'Original' filing first (what the market saw first). Halves download size/time."""
    rows = [f for f in rows if isinstance(f, dict) and str(f.get("xbrl", "")).lower().endswith(".xml")]
    sym = rows[0].get("symbol") if rows else None
    bases = {f.get("consolidated") for f in rows}
    basis = old_basis().get(sym)
    if basis not in bases:
        basis = "Consolidated" if "Consolidated" in bases else "Standalone"
    best = {}
    for f in rows:
        if f.get("consolidated") != basis:
            continue
        key = f.get("qe_Date")
        rank = (f.get("type_Sub") != "Original", pd.to_datetime(f.get("broadcast_Date"), errors="coerce", dayfirst=True))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, f)
    return [f for _, f in best.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true", help="re-fetch filing lists already saved")
    a = ap.parse_args()
    syms = universe()[: a.limit] if a.limit else universe()
    (BRONZE / "list").mkdir(parents=True, exist_ok=True)
    (BRONZE / "xbrl").mkdir(parents=True, exist_ok=True)
    logp = BRONZE / "log.csv"
    new_log = not logp.exists()
    fh = logp.open("a", newline="")
    w = csv.writer(fh)
    if new_log:
        w.writerow(["time", "symbol", "what", "status", "detail"])
    sess = requests.Session()
    sess.headers.update(HEADERS)
    n_list = n_xml = 0
    try:
        get(sess, PAGE)                                            # normal page visit (sets the site's cookies)
        for i, sym in enumerate(syms, 1):
            lp = BRONZE / "list" / f"{re.sub(r'[^A-Za-z0-9&_-]', '_', sym)}.json"
            if lp.exists() and not a.refresh:
                rows = json.loads(lp.read_text())
            else:
                r = get(sess, API, params={"index": "equities", "type": "Integrated Filing- Financials",
                                           "symbol": sym, "page": 1, "size": 50})
                d = r.json() if r.status_code == 200 else {}
                rows = d.get("data") if isinstance(d, dict) else d
                rows = rows if isinstance(rows, list) else []
                lp.write_text(json.dumps(rows))
                n_list += 1
                w.writerow([datetime.now().isoformat(timespec="seconds"), sym, "list", r.status_code, len(rows)])
            for f in pick_one_per_quarter(rows):
                url = f.get("xbrl")
                if not url or not str(url).lower().endswith(".xml"):
                    continue
                xp = BRONZE / "xbrl" / sym / str(url).rsplit("/", 1)[-1]
                if xp.exists():
                    continue
                r = get(sess, url)
                if r.status_code == 200 and r.content.lstrip()[:1] == b"<":
                    xp.parent.mkdir(parents=True, exist_ok=True)
                    xp.write_bytes(r.content)
                    n_xml += 1
                w.writerow([datetime.now().isoformat(timespec="seconds"), sym, "xbrl", r.status_code, xp.name])
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(syms)} companies | new lists {n_list} | new XBRL files {n_xml}", flush=True)
    except Blocked as e:
        print(f"STOPPED: {e}. Nothing is lost - run the same command again later (an hour or a day).")
        return 2
    finally:
        fh.close()
    print(f"done: {len(syms)} companies | new lists {n_list} | new XBRL files {n_xml} -> {BRONZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
