"""
Create a triton stub module for Windows compatibility
"""
import sys
import os
from pathlib import Path

# Get user site-packages directory
import site
user_site = site.getusersitepackages()
print(f"User site-packages: {user_site}")

# Create triton directory and __init__.py
triton_dir = Path(user_site) / "triton"
triton_dir.mkdir(exist_ok=True)

# Create __init__.py with a minimal stub
init_file = triton_dir / "__init__.py"
init_content = '''"""
Triton stub for Windows compatibility
Triton is not available on Windows, so this is a minimal stub
"""
__version__ = "0.0.0"

# Minimal API stubs that might be needed
class MockFunction:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Triton is not available on Windows")

# Common triton functions that might be imported
jit = MockFunction
autotune = MockFunction
heuristics = MockFunction
'''

with open(init_file, 'w') as f:
    f.write(init_content)

print(f"Created triton stub at: {init_file}")
print("You may need to restart Python for this to take effect.")

