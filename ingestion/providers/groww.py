"""Groww market-data adapter (historical + live quotes).

Only endpoints exposed by the official ``growwapi`` SDK (v1.5.0) are used:
``get_historical_candles``, ``get_expiries``, ``get_contracts``, ``get_quote``,
``get_ltp``, ``get_option_chain``, ``get_greeks`` and the instrument CSV.
Rate limits follow the documented per-category limits (see config/settings.yaml).
Order placement lives in ``execution/groww_adapter.py``; it is NOT retried here.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from core.clock import IST
from core.models import ChainRow, OptionChain, Quote
from core.ratelimit import RateLimiter
from ingestion.providers.base import CANDLE_COLUMNS, HistoricalDataProvider, LiveDataProvider, ProviderError

log = logging.getLogger(__name__)

INTERVAL_MINUTES = {"1minute": 1, "2minute": 2, "3minute": 3, "5minute": 5, "10minute": 10, "15minute": 15,
                    "30minute": 30, "1hour": 60, "4hour": 240, "1day": 1440}


def _retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in ("GrowwAPIRateLimitException", "GrowwAPITimeoutException"):
        return True
    if "rate limit" in str(exc).lower():  # Groww also reports rate limits as a generic GrowwAPIException
        return True
    if name in ("GrowwAPIAuthenticationException", "GrowwAPIAuthorisationException",
                "GrowwAPIBadRequestException", "GrowwAPINotFoundException"):
        return False
    import requests

    return isinstance(exc, (requests.ConnectionError, requests.Timeout, ValueError))  # ValueError: bad JSON on 5xx


def groww_option_symbol(underlying: str, expiry: date, strike: float, opt: str) -> str:
    """Groww symbol format observed from /historical/contracts, e.g. ``NSE-NIFTY-17Nov26-25000-CE``."""
    return f"NSE-{underlying}-{expiry.strftime('%d%b%y')}-{int(strike)}-{opt}"


def groww_future_symbol(underlying: str, expiry: date) -> str:
    return f"NSE-{underlying}-{expiry.strftime('%d%b%y')}-FUT"


class GrowwProvider(HistoricalDataProvider, LiveDataProvider):
    name = "groww"

    def __init__(self, settings: dict, client=None, token: Optional[str] = None, sleep: Callable = time.sleep):
        if client is None:
            from growwapi import GrowwAPI

            from core.secrets import obtain_access_token

            client = GrowwAPI(token or obtain_access_token())
        self.client = client
        rl = settings["rate_limits"]
        self.limiters = {
            "live": RateLimiter(rl["live_data_per_sec"], rl["live_data_per_min"], sleep=sleep),
            "non_trading": RateLimiter(rl["non_trading_per_sec"], rl["non_trading_per_min"], sleep=sleep),
            "historical": RateLimiter(rl["historical_per_sec"], rl["historical_per_min"], sleep=sleep),
        }
        h = settings.get("historical", {})
        self.max_days = h.get("max_days_per_request", {})
        self.max_retries = h.get("max_retries", 5)
        self.backoff = h.get("backoff_base_seconds", 1.0)
        self._sleep = sleep
        self.failures: list[dict] = []
        self.last_success_at: Optional[datetime] = None
        self.consecutive_failures = 0
        self.token_refreshes = 0
        self._instruments: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ plumbing
    def refresh_token(self) -> bool:
        """Re-authenticate via TOTP or key/secret (if configured). Returns True if a new token was installed."""
        from core.secrets import get_secret, obtain_access_token

        if not (get_secret("GROWW_API_KEY") and (get_secret("GROWW_TOTP_SECRET") or get_secret("GROWW_API_SECRET"))):
            return False
        import os

        os.environ.pop("GROWW_ACCESS_TOKEN", None)
        try:
            tok = obtain_access_token(force_generate=True)
        except Exception as e:  # noqa: BLE001
            log.error("token refresh failed: %s", type(e).__name__)
            return False
        self.client.token = tok
        self.token_refreshes += 1
        log.warning("access token refreshed")
        return True

    def _call(self, category: str, fn: Callable, *args, **kwargs):
        import time as _t

        from monitoring.metrics import METRICS

        last: Optional[Exception] = None
        refreshed = False
        for attempt in range(self.max_retries):
            self.limiters[category].acquire()
            t0 = _t.monotonic()
            try:
                out = fn(*args, **kwargs)
                METRICS.observe(f"groww_{category}", _t.monotonic() - t0)
                self.last_success_at = datetime.now(IST)
                self.consecutive_failures = 0
                return out
            except Exception as e:  # noqa: BLE001
                METRICS.inc(f"groww_{category}_errors")
                last = e
                if type(e).__name__ == "GrowwAPIAuthenticationException" and not refreshed:
                    refreshed = True
                    if self.refresh_token():
                        continue
                if not _retryable(e):
                    break
                wait = self.backoff * (2 ** attempt)
                log.warning("groww %s retry %d after %s: %s", getattr(fn, "__name__", fn), attempt + 1, type(e).__name__, e)
                self._sleep(wait)
        self.consecutive_failures += 1
        raise ProviderError(f"{getattr(fn, '__name__', fn)} failed: {type(last).__name__}: {last}") from last

    # ---------------------------------------------------------------- historical
    def get_historical_candles(self, instrument: str, start: datetime, end: datetime, interval: str,
                               segment: str = "CASH") -> pd.DataFrame:
        """Download candles, splitting into documented max-range windows.

        ``instrument`` is a Groww symbol (``NSE-NIFTY``, ``NSE-NIFTY-17Nov26-25000-CE``).
        Failed windows are recorded in ``self.failures`` and skipped (never substituted).
        """
        max_days = int(self.max_days.get(interval, 30))
        frames = []
        cur = start
        while cur <= end:
            w_end = min(end, cur + timedelta(days=max_days) - timedelta(seconds=1))
            try:
                resp = self._call("historical", self.client.get_historical_candles, "NSE", segment, instrument,
                                  cur.strftime("%Y-%m-%d %H:%M:%S"), w_end.strftime("%Y-%m-%d %H:%M:%S"), interval)
                frames.append(self.candles_to_frame(resp.get("candles") or []))
            except ProviderError as e:
                self.failures.append({"instrument": instrument, "interval": interval, "start": cur.isoformat(),
                                      "end": w_end.isoformat(), "error": str(e)})
                log.error("candle window failed %s %s..%s: %s", instrument, cur, w_end, e)
            cur = w_end + timedelta(seconds=1)
        if not frames:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def candles_to_frame(candles: list) -> pd.DataFrame:
        rows = []
        for c in candles:
            c = list(c) + [None] * (7 - len(c))
            rows.append(c[:7])
        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
        if df.empty:
            return df
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(IST)
        else:
            df["timestamp"] = pd.to_datetime(ts).dt.tz_localize(IST)
        for col in CANDLE_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_instruments(self) -> pd.DataFrame:
        if self._instruments is None:
            self._instruments = self._call("non_trading", self.client.get_all_instruments)
        return self._instruments

    def get_expiries(self, underlying: str, year: int, month: Optional[int] = None) -> list[date]:
        r = self._call("historical", self.client.get_expiries, "NSE", underlying, year, month)
        return sorted(date.fromisoformat(x) for x in r.get("expiries", []))

    def get_option_contracts(self, underlying: str, expiry: date) -> list[str]:
        r = self._call("historical", self.client.get_contracts, "NSE", underlying, expiry.isoformat())
        return list(r.get("contracts", []))

    def futures_contracts(self, underlying: str) -> pd.DataFrame:
        df = self.get_instruments()
        f = df[(df.underlying_symbol == underlying) & (df.instrument_type == "FUT") & (df.exchange == "NSE")].copy()
        f["expiry_date"] = pd.to_datetime(f["expiry_date"]).dt.date
        return f.sort_values("expiry_date")

    def option_instruments(self, underlying: str, expiry: date) -> pd.DataFrame:
        df = self.get_instruments()
        o = df[(df.underlying_symbol == underlying) & (df.instrument_type.isin(["CE", "PE"])) & (df.exchange == "NSE")].copy()
        o["expiry_date"] = pd.to_datetime(o["expiry_date"]).dt.date
        return o[o.expiry_date == expiry]

    # ---------------------------------------------------------------------- live
    def get_quote(self, trading_symbol: str, segment: str) -> Quote:
        r = self._call("live", self.client.get_quote, trading_symbol, "NSE", segment)
        depth = r.get("depth") or {}
        buy = (depth.get("buy") or [{}])[0] or {}
        sell = (depth.get("sell") or [{}])[0] or {}
        bid = buy.get("price") or r.get("bid_price")
        ask = sell.get("price") or r.get("offer_price")
        ltt = r.get("last_trade_time")
        if ltt:
            ltt = datetime.fromtimestamp(ltt / 1000 if ltt > 1e11 else ltt, IST)
        return Quote(trading_symbol=trading_symbol, ltp=float(r.get("last_price") or 0.0),
                     bid=float(bid) if bid else None, ask=float(ask) if ask else None,
                     bid_qty=buy.get("quantity"), ask_qty=sell.get("quantity"),
                     volume=r.get("volume"), open_interest=r.get("open_interest"),
                     last_trade_time=ltt, received_at=datetime.now(IST), raw=r)

    def get_ltp(self, symbols: list[str], segment: str) -> dict[str, float]:
        return self._call("live", self.client.get_ltp, tuple(symbols), segment)

    def get_option_chain(self, underlying: str, expiry: date) -> OptionChain:
        r = self._call("live", self.client.get_option_chain, "NSE", underlying, expiry.isoformat())
        rows: list[ChainRow] = []
        for strike, sides in (r.get("strikes") or {}).items():
            for ot in ("CE", "PE"):
                s = sides.get(ot)
                if not s:
                    continue
                g = s.get("greeks") or {}
                rows.append(ChainRow(strike=float(strike), option_type=ot, trading_symbol=s.get("trading_symbol"),
                                     ltp=float(s.get("ltp") or 0), open_interest=float(s.get("open_interest") or 0),
                                     volume=float(s.get("volume") or 0), iv=g.get("iv"), delta=g.get("delta"),
                                     gamma=g.get("gamma"), theta=g.get("theta"), vega=g.get("vega"), rho=g.get("rho")))
        return OptionChain(underlying=underlying, expiry=expiry, underlying_ltp=float(r.get("underlying_ltp") or 0),
                           rows=rows, fetched_at=datetime.now(IST))

    def get_available_option_buy_balance(self) -> float:
        r = self._call("non_trading", self.client.get_available_margin_details)
        return float((r.get("fno_margin_details") or {}).get("option_buy_balance_available") or 0.0)
