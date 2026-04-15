import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class FeatureCache:
    def __init__(self, root: Path, fingerprint: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def _path(self, sequence: str, kind: str) -> Path:
        h = hashlib.sha256(
            (self.fingerprint + "\n" + kind + "\n" + sequence.upper()).encode()
        ).hexdigest()
        return self.root / f"{h}.npz"

    def load_baseline(self, sequence: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        p = self._path(sequence, "baseline")
        if not p.is_file():
            return None
        z = np.load(p)
        return z["bloom"], z["dnabert"]

    def save_baseline(
        self, sequence: str, bloom: np.ndarray, dnabert: np.ndarray
    ) -> None:
        p = self._path(sequence, "baseline")
        np.savez_compressed(p, bloom=bloom, dnabert=dnabert)

    def load_bgpca(
        self, sequence: str, max_tokens: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        p = self._path(sequence, f"bgpca_{max_tokens}")
        if not p.is_file():
            return None
        z = np.load(p)
        return z["hidden"], z["bloom_sig"], z["bloom_sum"]

    def save_bgpca(
        self,
        sequence: str,
        max_tokens: int,
        hidden: np.ndarray,
        bloom_sig: np.ndarray,
        bloom_sum: np.ndarray,
    ) -> None:
        p = self._path(sequence, f"bgpca_{max_tokens}")
        np.savez_compressed(
            p, hidden=hidden, bloom_sig=bloom_sig, bloom_sum=bloom_sum
        )
