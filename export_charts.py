import json
from pathlib import Path
import urllib.request

BASE = "http://127.0.0.1:8000"
OUT = Path("frontend/data")
OUT.mkdir(parents=True, exist_ok=True)

def get_json(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

# Try common chart endpoints used by the project
candidates = [
    "/charts/distributions",
    "/charts",
    "/metrics/charts",
]

last_error = None
data = None
for path in candidates:
    try:
        data = get_json(path)
        print(f"Using endpoint: {path}")
        break
    except Exception as e:
        last_error = e
        print(f"Failed {path}: {e}")

if data is None:
    raise SystemExit(f"Could not fetch chart data. Last error: {last_error}")

(OUT / "charts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Saved frontend/data/charts.json")
print(json.dumps(data, indent=2)[:1000])