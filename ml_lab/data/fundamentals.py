"""Quarterly financial results of Indian companies from NSE corporate filings (point-in-time).

Source: NSE corporate filings (www.nseindia.com, "Financial Results").  For every company we take the list of filed
results, then the reported numbers of each STANDALONE QUARTERLY result (one JSON per filing; same field names for the
old 2007-2018 format and the new XBRL-era format, banks included).  Every row keeps its filing time, so a number is
only "known" after it was published -> no look-ahead when used as a model feature.

Steps (resumable; raw JSON kept in data_lake/bronze/nse_results/):
    python3 -m ml_lab.data.fundamentals symbols   # NSE symbols of every stock ever in the liquid universe
    python3 -m ml_lab.data.fundamentals list      # filing list per company
    python3 -m ml_lab.data.fundamentals detail    # reported numbers per filing (slow: ~100k requests, polite rate)
    python3 -m ml_lab.data.fundamentals build     # -> data_lake/silver/fundamentals/quarterly.parquet
    python3 -m ml_lab.data.fundamentals all

Public data, fetched slowly for personal research.  Amounts are in Rs lakh as filed.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests

from core.config import load_config, resolve_path

LAKE = resolve_path(load_config(), "data_lake")
RAW = LAKE / "bronze" / "nse_results"
LIST_DIR, DET_DIR = RAW / "list", RAW / "detail"
OUT = LAKE / "silver" / "fundamentals"
FROM_PERIOD = pd.Timestamp("2010-04-01")          # need one year before 2011 for YoY growth
BASE = "https://www.nseindia.com/api/"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"}
WORKERS, PAUSE = 3, 0.35

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
    return _local.s


def _get(url: str, tries: int = 5):
    for k in range(tries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 200:
                time.sleep(PAUSE)
                return r.json()
            if r.status_code in (401, 403, 429):              # refused / rate-limited: stop, never work around it
                raise SystemExit(f"NSE refused the request (HTTP {r.status_code}); stopping. Resume later.")
            if r.status_code >= 500:
                time.sleep(10 * (k + 1))
                continue
            return None
        except (requests.RequestException, ValueError):
            time.sleep(5 * (k + 1))
    return None


# ------------------------------------------------------------------ steps
def symbols() -> list[str]:
    """NSE symbols of every stock that was in the liquid universe (price >= Rs 20, >= Rs 2 cr/day) since 2010."""
    p = RAW / "symbols.json"
    if p.exists():
        return json.loads(p.read_text())
    from ml_lab.indicator_backtest import Data
    d = Data()
    syms = sorted(s for s in d.C.columns if not s.startswith("BSE:"))
    RAW.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(syms))
    print(f"{len(syms)} NSE symbols ({len(d.C.columns) - len(syms)} BSE-only stocks have no NSE filings)", flush=True)
    return syms


def fetch_lists(syms: list[str]) -> None:
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in syms if not (LIST_DIR / f"{s}.json").exists()]
    print(f"filing lists: {len(syms) - len(todo)} done, {len(todo)} to fetch", flush=True)

    def one(s):
        from urllib.parse import quote
        d = _get(BASE + f"corporates-financial-results?index=equities&period=Quarterly&symbol={quote(s)}")
        if d is not None:
            (LIST_DIR / f"{s}.json").write_text(json.dumps(d))
        return s, d is not None

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(one, s) for s in todo]), 1):
            if i % 100 == 0:
                print(f"  lists {i}/{len(todo)}", flush=True)


def wanted_filings() -> pd.DataFrame:
    rows = []
    for p in LIST_DIR.glob("*.json"):
        for x in json.loads(p.read_text()):
            rows.append(x)
    F = pd.DataFrame(rows)
    if F.empty:
        return F
    F["period_end"] = pd.to_datetime(F.toDate, format="%d-%b-%Y", errors="coerce")
    F = F[(F.period == "Quarterly") & F.consolidated.str.startswith("Non") &
          F.cumulative.str.startswith("Non") & (F.period_end >= FROM_PERIOD)]
    return F.drop_duplicates("seqNumber")


def fetch_details() -> None:
    DET_DIR.mkdir(parents=True, exist_ok=True)
    F = wanted_filings()
    todo = F[[not (DET_DIR / f"{s}.json").exists() for s in F.seqNumber]]
    print(f"filings: {len(F)} wanted, {len(F) - len(todo)} done, {len(todo)} to fetch", flush=True)
    t0 = time.time()

    def one(x):
        u = (BASE + "corporates-financial-results-data?index=equities&params=%s&seq_id=%s&industry=%s"
             "&frOldNewFlag=%s&ind=%s&format=%s" % (x.params, x.seqNumber, x.industry, x.oldNewFlag, x.reInd, x.format))
        d = _get(u)
        if d is not None:
            (DET_DIR / f"{x.seqNumber}.json").write_text(json.dumps(d))
        return d is not None

    ok = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, x) for x in todo.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            ok += f.result()
            if i % 1000 == 0:
                rate = i / (time.time() - t0)
                print(f"  details {i}/{len(todo)} ({ok} ok) {rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left", flush=True)


# fields as filed (Rs lakh); first non-empty candidate wins
FIELDS = {
    "sales": ["re_net_sale", "re_rev_frm_opr", "re_revenue_from_operations", "re_gro_inc"],
    "total_income": ["re_total_inc", "re_tot_inc", "re_gro_inc"],
    "other_income": ["re_oth_inc_new", "re_oth_inc"],
    "net_profit": ["re_net_profit", "re_con_pro_loss", "re_proloss_ord_act"],
    "pbt": ["re_pro_loss_bef_tax"],
    "interest": ["re_int_new", "re_fin_cost", "re_int"],
    "depreciation": ["re_depr_und_exp"],
    "eps": ["re_basic_eps", "re_basic_eps_for_cont_dic_opr", "re_bsc_eps_bfr_exi"],
    "paid_up": ["re_pdup"],
    "face_value": ["re_face_val"],
    # banks
    "interest_earned": ["re_int_earned"],
    "interest_expended": ["re_int_expd"],
    "gross_npa_pct": ["re_grs_npa_per"],
}


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def build() -> pd.DataFrame:
    F = wanted_filings().set_index("seqNumber")
    rows = []
    for p in DET_DIR.glob("*.json"):
        seq = p.stem
        if seq not in F.index:
            continue
        j = json.loads(p.read_text())
        rd = j.get("resultsData2") or j.get("resultsData") or {}
        if not isinstance(rd, dict):
            continue
        f = F.loc[seq]
        row = {"sym": f.symbol, "isin": f.isin, "period_end": f.period_end, "bank": f.bank == "B",
               "filed": pd.to_datetime(f.filingDate if f.filingDate != "-" else None, format="%d-%b-%Y %H:%M", errors="coerce"),
               "audited": f.audited, "seq": int(seq)}
        for k, cands in FIELDS.items():
            row[k] = next((_num(rd[c]) for c in cands if _num(rd.get(c)) is not None), None)
        rows.append(row)
    Q = pd.DataFrame(rows)
    # several filings for one quarter (revisions): keep the FIRST one filed -> what the market saw first
    Q = Q.dropna(subset=["filed"]).sort_values(["sym", "period_end", "filed"]).drop_duplicates(["sym", "period_end"])
    Q["shares"] = Q.paid_up * 1e5 / Q.face_value                       # paid-up capital (Rs lakh) / face value
    OUT.mkdir(parents=True, exist_ok=True)
    Q.to_parquet(OUT / "quarterly.parquet", index=False)
    cov = {"at": datetime.now().isoformat(timespec="seconds"), "companies": int(Q.sym.nunique()), "rows": len(Q),
           "period_from": str(Q.period_end.min().date()), "period_to": str(Q.period_end.max().date()),
           "filled_pct": {k: round(float(Q[k].notna().mean() * 100), 1) for k in FIELDS}}
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps(cov, indent=1))
    return Q


def main(argv: list[str]) -> int:
    step = argv[0] if argv else "all"
    if step in ("symbols", "list", "all"):
        syms = symbols()
        if step != "symbols":
            fetch_lists(syms)
    if step in ("detail", "all"):
        fetch_details()
    if step in ("build", "all"):
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
