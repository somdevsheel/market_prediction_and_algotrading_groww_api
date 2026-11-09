# Unattended automation (Linux, systemd user services)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/nifty-scheduler.service deploy/nifty-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nifty-dashboard nifty-scheduler
loginctl enable-linger "$USER"        # keep services running when you are logged out
journalctl --user -u nifty-scheduler -f   # follow the logs
systemctl --user stop nifty-scheduler     # stop automation
```
The scheduler service runs `--mode live`: each trading morning at 09:05 it starts LIVE only if every check in
`monitoring/live_gate.py` passes, otherwise it runs PAPER and raises a `LIVE_GATE_FAILED` alert.
The computer must be on, online and on a Groww-whitelisted IP from ~08:45 to 15:45 IST on trading days.
