"""Point-in-time quarterly fundamentals panel for NSE stocks (copied from the Quantera data lake).

Inputs (copied 2026-11-18 from ~/Project/Quantera, NSE financial-results XBRL filings, periods 2017-03 .. 2024-12)
  + 2025+ quarters from NSE Integrated Filing (python3 -m ml_lab.data.nse_integrated_filings, parsed here, cached):
  data_lake/gold/india_fundamentals/nse_key_financials.parquet      quarterly P&L, one row per filing
  data_lake/silver/india_fundamentals/xbrl_facts/symbol=*/*.parquet  every XBRL fact (balance sheet items come from here)

Output: data_lake/gold/india_fundamentals/fund_panel.parquet - one row per (symbol, quarter end) with
  avail      first trading day the numbers may be used = filing date + 1 day (results usually come after market hours)
  rev, np, eps, shares                         that quarter (P&L); shares = paid-up capital / face value
  rev_ttm, np_ttm, eps_ttm                     last 4 quarters (only when all 4 exist)
  rev_yoy, np_yoy, margin, margin_chg, sue     growth vs same quarter last year; SUE = YoY EPS change / std of last 8
  equity, debt, cash, assets                   latest balance sheet published up to this filing (half-yearly in India)
  split_after                                  share-count jumps >= 1.45x after this quarter (bonus/split) - used to put
                                               split-adjusted Yahoo prices back into the money of that date
One basis per company: consolidated where the company files consolidated results for most quarters, else standalone.

    python3 -m ml_lab.data.india_fundamentals
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from core.config import load_config, resolve_path

LAKE = resolve_path(load_config(), "data_lake")
GOLD = LAKE / "gold" / "india_fundamentals"
SILVER = LAKE / "silver" / "india_fundamentals" / "xbrl_facts"
OUT = GOLD / "fund_panel.parquet"
INTEGRATED = LAKE / "bronze" / "india_fundamentals" / "integrated"   # 2025+ filings (ml_lab.data.nse_integrated_filings)
INTEGRATED_CACHE = LAKE / "silver" / "india_fundamentals" / "integrated_parsed.parquet"
BS_CONCEPTS = {"Equity": "equity", "EquityAttributableToOwnersOfParent": "equity_owners",
               "BorrowingsNoncurrent": "debt_nc", "BorrowingsCurrent": "debt_c", "Borrowings": "debt_bank",
               "CashAndCashEquivalents": "cash", "Assets": "assets",
               "Capital": "bank_capital", "ReservesAndSurplus": "bank_reserves",           # banks' equity
               "CashAndBalancesWithReserveBankOfIndia": "bank_cash"}
# first concept found wins (same lists as the Quantera gold table, plus the bank-format names)
P_CONCEPTS = {
    "rev": ["RevenueFromOperations"],
    "total_income": ["Income", "TotalIncome"],
    "np_owners": ["ProfitOrLossAttributableToOwnersOfParent"],
    "np": ["ProfitLossForPeriod", "ProfitLossForThePeriod", "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates"],
    "eps": ["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations", "BasicEarningsLossPerShareFromContinuingOperations",
            "BasicEarningsPerShare", "BasicEarningsPerShareAfterExtraordinaryItems"],
    "paid_up": ["PaidUpValueOfEquityShareCapital"],
    "face": ["FaceValueOfEquityShareCapital"],
}


def _pnl() -> pd.DataFrame:
    g = pd.read_parquet(GOLD / "nse_key_financials.parquet")
    g = g[(g.period_type == "Quarterly")].copy()
    for c in ("from_date", "to_date", "filing_datetime"):
        g[c] = pd.to_datetime(g[c])
    g = g[(g.to_date - g.from_date).dt.days.between(85, 95)]
    g["np"] = g.net_profit_attributable_to_owners.fillna(g.net_profit)
    g["shares"] = g.paid_up_equity_capital / g.face_value_per_share
    g["rev"] = g.revenue_from_operations.fillna(g.total_income)        # banks report total income, not revenue
    g = g.rename(columns={"eps_basic": "eps"})
    cols = ["symbol", "isin", "to_date", "filing_datetime", "consolidated", "rev", "np", "eps", "shares"]
    new = _integrated()
    if len(new):
        g = pd.concat([g[cols], new[cols + ["equity", "debt", "cash", "assets"]]], ignore_index=True)
    # first filing of each quarter = what the market saw first (later revisions would be look-ahead)
    g = g.sort_values("filing_datetime").drop_duplicates(["symbol", "to_date", "consolidated"], keep="first")
    cons = g.groupby("symbol").consolidated.apply(lambda s: (s == "Consolidated").mean())
    pick = cons.map(lambda x: "Consolidated" if x >= 0.5 else "Non-Consolidated")
    g = g[g.consolidated == g.symbol.map(pick)]
    g["eps"] = g.eps.fillna(g.np / g.shares)                           # banks: no EPS tag in the old format
    return g[[c for c in g.columns if c in cols + ["equity", "debt", "cash", "assets"]]]


def _parse_one(path: str) -> dict:
    """One integrated-filing XBRL -> the quarter's P&L (3-month context) and balance sheet (instant), no dimensions."""
    from lxml import etree
    X = "{http://www.xbrl.org/2003/instance}"
    root = etree.parse(path).getroot()
    ctx = {}
    for c in root.iter(X + "context"):
        seg = c.find(f"{X}entity/{X}segment")
        if seg is not None and len(seg):
            continue
        p = c.find(X + "period")
        ctx[c.get("id")] = (p.findtext(X + "startDate"), p.findtext(X + "endDate"), p.findtext(X + "instant"))
    dur, inst = {}, {}
    for e in root:
        if not isinstance(e.tag, str) or e.get("contextRef") not in ctx:
            continue
        try:
            v = float(e.text)
        except (TypeError, ValueError):
            continue
        s, en, i = ctx[e.get("contextRef")]
        name = etree.QName(e).localname
        if i:
            inst.setdefault((name, i), v)
        elif s and en:
            dur.setdefault((name, s, en), v)
    return {"dur": dur, "inst": inst}


def _integrated() -> pd.DataFrame:
    """Rows in the same shape as the old panel input, from the 2025+ integrated filings (cached by file name)."""
    lists = sorted((INTEGRATED / "list").glob("*.json")) if INTEGRATED.exists() else []
    if not lists:
        return pd.DataFrame()
    import json
    from concurrent.futures import ProcessPoolExecutor

    from ml_lab.data.nse_integrated_filings import pick_one_per_quarter
    meta = []
    for lp in lists:
        for f in pick_one_per_quarter(json.loads(lp.read_text())):
            path = INTEGRATED / "xbrl" / f["symbol"] / f["xbrl"].rsplit("/", 1)[-1]
            if path.exists():
                meta.append({"symbol": f["symbol"], "file": path.name, "path": str(path),
                             "to_date": pd.to_datetime(f["qe_Date"], format="%d-%b-%Y"),
                             "filing_datetime": pd.to_datetime(f["broadcast_Date"], format="%d-%b-%Y %H:%M:%S", errors="coerce"),
                             "consolidated": "Consolidated" if f["consolidated"] == "Consolidated" else "Non-Consolidated"})
    m = pd.DataFrame(meta)
    if m.empty:
        return m
    cache = pd.read_parquet(INTEGRATED_CACHE) if INTEGRATED_CACHE.exists() else pd.DataFrame(columns=["file"])
    todo = m[~m.file.isin(set(cache.file))]
    if len(todo):
        rows = []
        with ProcessPoolExecutor(max_workers=6) as ex:
            for (_, r), res in zip(todo.iterrows(), ex.map(_parse_one, todo.path, chunksize=20)):
                q_end = r.to_date.strftime("%Y-%m-%d")
                out = {"file": r.file}
                durs = {(n, s): v for (n, s, e), v in res["dur"].items()
                        if e == q_end and 85 <= (pd.Timestamp(e) - pd.Timestamp(s)).days <= 95}   # the 3-month quarter
                anyd = {n: v for (n, s, e), v in res["dur"].items() if e == q_end}
                for k, names in P_CONCEPTS.items():
                    src = anyd if k in ("paid_up", "face") else {n: v for (n, s), v in durs.items()}
                    out[k] = next((src[n] for n in names if n in src), np.nan)
                ins = {n: v for (n, i), v in res["inst"].items() if i == q_end}
                for n, k in BS_CONCEPTS.items():
                    out[k] = ins.get(n, np.nan)
                rows.append(out)
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        INTEGRATED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(INTEGRATED_CACHE, index=False)
    d = m.merge(cache, on="file", how="left")
    d["np"] = d.np_owners.fillna(d.np)
    d["rev"] = d.rev.fillna(d.total_income)
    d["shares"] = d.paid_up / d.face
    d["isin"] = None
    _bs_combine(d)
    return d


def _bs_combine(w: pd.DataFrame) -> None:
    for c in BS_CONCEPTS.values():
        if c not in w:
            w[c] = np.nan
    w["equity"] = w.equity_owners.fillna(w.equity).fillna(w.bank_capital + w.bank_reserves)
    w["debt"] = w[["debt_nc", "debt_c"]].sum(axis=1, min_count=1).fillna(w.debt_bank)
    w["cash"] = w.cash.fillna(w.bank_cash)


def _balance_sheet() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SILVER / "symbol=*" / "*.parquet")))
    d = ds.dataset(fs, format="parquet", schema=pq.read_schema(fs[0]))
    f = ds.field("concept").isin(list(BS_CONCEPTS)) & (ds.field("dimension_count") == 0) & (ds.field("period_type") == "instant")
    t = d.to_table(columns=["symbol", "concept", "period_end", "to_date", "numeric_value", "consolidated", "filing_datetime"],
                   filter=f).to_pandas()
    t["period_end"] = pd.to_datetime(t.period_end, errors="coerce")
    t["to_date"] = pd.to_datetime(t.to_date, format="%d-%b-%Y", errors="coerce")
    t["filing_datetime"] = pd.to_datetime(t.filing_datetime, errors="coerce")
    t = t[t.period_end == t.to_date]                                   # balance sheet of this filing's period end
    t = t.sort_values("filing_datetime").drop_duplicates(["symbol", "consolidated", "to_date", "concept"], keep="first")
    w = t.pivot_table(index=["symbol", "consolidated", "to_date"], columns="concept", values="numeric_value", aggfunc="first")
    w = w.rename(columns=BS_CONCEPTS).reset_index()
    _bs_combine(w)
    return w[["symbol", "consolidated", "to_date", "equity", "debt", "cash", "assets"]]


def _clean_shares(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share counts in the filings have unit slips (lakhs / crores for one quarter -> 1e4..1e7 jumps that revert).
    Accept a value when it is within 0.9x..1.45x of the running count (issuance, buyback); accept a jump of
    1.45x..25x (bonus / split) or 0.04x..0.9x (consolidation) only when the next filed quarter confirms it;
    otherwise keep the running count. Returns (clean shares, split factor per quarter; 1 = no split)."""
    out, k, cur = np.full(len(v), np.nan), np.ones(len(v)), np.nan
    valid = np.where(np.isfinite(v) & (v > 0))[0]
    for n, i in enumerate(valid):
        x = v[i]
        if not np.isfinite(cur):
            cur = x
        else:
            r = x / cur
            nxt = v[valid[n + 1]] if n + 1 < len(valid) else np.nan
            confirmed = np.isfinite(nxt) and 0.8 <= nxt / x <= 1.25
            if 0.9 <= r < 1.45:
                cur = x
            elif (1.45 <= r <= 25 or 0.04 <= r < 0.9) and confirmed:
                k[i], cur = r, x
        out[i] = cur
    return out, k


def build_panel() -> pd.DataFrame:
    g = _pnl()
    bs = _balance_sheet()
    g = g.merge(bs, on=["symbol", "consolidated", "to_date"], how="left", suffixes=("", "_old"))
    for c in ("equity", "debt", "cash", "assets"):                     # 2025+ rows carry their own balance sheet
        if f"{c}_old" in g:
            g[c] = g[c].fillna(g[f"{c}_old"]) if c in g else g[f"{c}_old"]
            g = g.drop(columns=f"{c}_old")
    g = g.sort_values(["symbol", "to_date"]).reset_index(drop=True)
    out = []
    for sym, x in g.groupby("symbol", sort=False):
        x = x.set_index("to_date").copy()
        q = x.index.to_period("Q")
        x = x[~q.duplicated(keep="first")]
        q = x.index.to_period("Q")
        full = pd.period_range(q.min(), q.max(), freq="Q")
        y = x.set_index(q).reindex(full)                          # gaps stay NaN, so lags are true calendar lags
        for c in ("rev", "np", "eps"):
            y[f"{c}_ttm"] = y[c].rolling(4, min_periods=4).sum()
            y[f"{c}_ly"] = y[c].shift(4)
        y["rev_yoy"] = y.rev / y.rev_ly - 1
        y["np_yoy"] = (y.np - y.np_ly) / y.np_ly.abs()
        y["margin"] = y.np / y.rev
        y["margin_chg"] = y.margin - (y.np_ly / y.rev_ly)
        for c in ("equity", "debt", "cash", "assets"):            # carry the last published balance sheet forward
            y[c] = y[c].ffill(limit=3)
        y["shares"], k = _clean_shares(y.shares.to_numpy())
        y["split_after"] = pd.Series(k[::-1], index=y.index[::-1]).cumprod()[::-1].shift(-1).fillna(1.0)   # jumps in LATER quarters
        # SUE on split-adjusted EPS (a bonus halves EPS; dividing every quarter by the later split factor puts all
        # quarters in the same share units - a common scale, so SUE (scale-free) uses nothing from the future)
        ea = y.eps / y.split_after
        de = ea - ea.shift(4)
        y["sue"] = de / de.rolling(8, min_periods=4).std().shift(1)
        y = y[y.symbol.notna()]
        y["to_date"] = y.index.to_timestamp(how="end").normalize()
        out.append(y.reset_index(drop=True))
    p = pd.concat(out, ignore_index=True)
    p["avail"] = p.filing_datetime.dt.normalize() + pd.Timedelta(days=1)
    for c in ("rev_yoy", "np_yoy", "sue", "margin_chg"):
        p[c] = p[c].replace([np.inf, -np.inf], np.nan)
    cols = ["symbol", "isin", "consolidated", "to_date", "filing_datetime", "avail", "rev", "np", "eps", "shares",
            "rev_ttm", "np_ttm", "eps_ttm", "rev_yoy", "np_yoy", "margin", "margin_chg", "sue",
            "equity", "debt", "cash", "assets", "split_after"]
    return p[cols]


def main() -> int:
    p = build_panel()
    p.to_parquet(OUT, index=False)
    print(f"{len(p):,} company-quarters | {p.symbol.nunique():,} companies | quarters {p.to_date.min().date()} .. "
          f"{p.to_date.max().date()} | last filing {p.filing_datetime.max().date()}")
    print("filled %:", {c: round(p[c].notna().mean() * 100) for c in ("rev_yoy", "np_yoy", "sue", "np_ttm", "equity", "debt")})
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
