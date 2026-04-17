"""
Remove legacy Triton shims and ensure DNABERT uses PyTorch attention (no Flash import).

The project no longer installs a fake ``triton`` package — that caused
``No module named 'triton.backends'`` when PyTorch probed a real Triton layout.

Run (optional, after upgrades):

    python create_triton_stub.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bloom_seq.plugins.dnabert2.triton_compat import prepare_dnabert_environment  # noqa: E402


def main() -> None:
    prepare_dnabert_environment()
    print("DNABERT environment prepared: shim triton removed, bert_layers Flash import disabled.")


if __name__ == "__main__":
    main()
