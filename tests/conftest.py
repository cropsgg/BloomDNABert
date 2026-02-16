"""
Test configuration and fixtures.

Mocks the `transformers` package so tests can run without installing
the full DNABERT-2 model weights (multi-GB download). Tests that
actually need the DNABERT model should be marked with
@pytest.mark.requires_dnabert and run separately.
"""

import sys
import types
from unittest.mock import MagicMock

# Mock heavy dependencies that may not be installed in the test environment
_OPTIONAL_DEPS = [
    'transformers',
    'plotly',
    'plotly.graph_objects',
    'plotly.subplots',
    'gradio',
]

for mod_name in _OPTIONAL_DEPS:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
