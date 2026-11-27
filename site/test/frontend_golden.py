"""Salida de es/frontend_es.py para cada línea de un archivo, en JSON (lo usa frontend_golden.mjs)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "es"))
import frontend_es  # noqa: E402

out = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").split("\n"):
    if not line:
        continue
    try:
        p = frontend_es.phonemize(line)
        out.append({"symbols": list(p.symbols), "ids": list(p.ids), "stress": list(p.stress),
                    "unknown": list(p.unknown)})
    except ValueError as error:
        out.append({"error": str(error)})
sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
