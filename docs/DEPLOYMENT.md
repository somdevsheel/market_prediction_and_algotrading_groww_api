# Deployment

Local (current): Python 3.10+ with deps from `pyproject.toml`.
```
python3 -m pip install --user growwapi==1.5.0 pandas numpy scipy pyarrow duckdb pyyaml fastapi uvicorn pyotp pytest httpx
python3 -m pytest
python3 -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
python3 scripts/run_trader.py
```
Daily automation: `python3 apps/worker/scheduler.py` (pre-market check 08:50, worker 09:05, EOD ingest + IV +
features + backup 15:45, trading days only). Keep it running in a terminal, tmux or a systemd user service.
Research: `python3 scripts/bulk_ingest.py …`, `python3 scripts/build_gold.py iv|features|ml`,
`python3 scripts/backtest_strategy.py --capital 20000 --risk 1000` (archived research runners: `archive/scripts/`).

Docker: `docker compose up api` and `docker compose run worker` (state, data lake and logs are host volumes).
Keep the API bound to 127.0.0.1 — it has no authentication and exposes the kill switch.

Secrets: `.env` (git-ignored). Prefer `GROWW_API_KEY` + `GROWW_TOTP_SECRET` for unattended mornings.
Static IP: Groww tokens carry allowed source IPs; run the worker from a whitelisted IP.
Backups: `python3 scripts/backup.py` → `backups/<ts>/` (online SQLite snapshot, governance, models, config,
manifests, gold) with SHA-256 manifest, integrity check and row-count verification; keeps 14. Set
`BACKUP_S3_BUCKET` (and install boto3 + AWS credentials) to also upload and verify on S3.
Alerts: set `ALERT_WEBHOOK_URL` in `.env` (Slack/Discord/Teams incoming webhook) to receive alerts.
Metrics: Prometheus can scrape `http://127.0.0.1:8100/metrics`.
