import json
from pathlib import Path
from typing import Dict

_CODON_PATH = Path(__file__).resolve().parent / "data" / "codon_table.json"


def load_codon_table(path: Path | None = None) -> Dict[str, str]:
    p = path or _CODON_PATH
    return json.loads(p.read_text(encoding="utf-8"))
