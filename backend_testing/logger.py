import csv, time
from pathlib import Path
LOG = Path("runs.csv")
HEADERS = "ts query answer fallback latency_ms".split()

def log_row(**kw):
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        if LOG.stat().st_size == 0:
            w.writeheader()
        w.writerow({h: kw.get(h, "") for h in HEADERS})
