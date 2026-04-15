import hashlib
from pathlib import Path


def load_fasta_sequence(fasta_path: Path) -> tuple[str, str]:
    text = fasta_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty FASTA: {fasta_path}")
    header = ""
    seq_parts: list[str] = []
    for ln in lines:
        if ln.startswith(">"):
            header = ln[1:].strip()
        else:
            seq_parts.append(ln.upper().replace(" ", ""))
    seq = "".join(seq_parts)
    if not seq:
        raise ValueError(f"No sequence in FASTA: {fasta_path}")
    allowed = set("ATCGN")
    bad = set(seq.upper()) - allowed
    if bad:
        raise ValueError(f"Invalid bases in {fasta_path}: {bad}")
    return header, seq.upper()


def verify_file_sha256(path: Path, expected_hex: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest().lower()
    exp = expected_hex.lower().strip()
    if got != exp:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {exp}, got {got}"
        )
