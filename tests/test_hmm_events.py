from datetime import date

import numpy as np

from ingestion.macro.events import EventCalendar
from quant.regime.hmm import GaussianHMM


def test_hmm_recovers_two_regimes():
    rng = np.random.default_rng(0)
    states = np.repeat([0, 1, 0, 1, 0], 200)
    x = np.where(states == 0, rng.normal(-1.0, 0.5, len(states)), rng.normal(2.0, 0.5, len(states)))
    m = GaussianHMM(n_states=2, seed=1).fit(x)
    p = m.filtered_probs(x)
    acc = (p.argmax(1) == states).mean()
    assert acc > 0.95
    assert m.transmat[0, 0] > 0.95 and m.transmat[1, 1] > 0.95
    assert m.describe()[0]["mean"][0] < m.describe()[1]["mean"][0]


def test_hmm_filtered_probs_are_causal():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (300, 2))
    m = GaussianHMM(n_states=3).fit(x)
    a = m.filtered_probs(x[:200])
    b = m.filtered_probs(np.vstack([x[:200], rng.normal(5, 1, (100, 2))]))
    assert np.allclose(a, b[:200])
    assert np.allclose(m.predict_next(x).sum(), 1)


def test_event_calendar():
    ev = EventCalendar()
    assert "rbi_policy" in ev.risk(date(2026, 12, 4))
    assert "budget" in ev.risk(date(2026, 2, 1))
    assert "cpi" in ev.risk(date(2026, 11, 13))  # 12th Thu after close -> Fri session
    assert ev.risk(date(2026, 11, 11)) == []
