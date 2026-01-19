"""
Launch the Bloom-Enhanced DNABERT Dashboard

This script starts the Gradio web interface for analyzing DNA sequences
for pathogenic variants, with a focus on sickle cell disease.

Usage:
    python launch_dashboard.py

The dashboard will be available at: http://localhost:7860
"""

import sys
import os

print("\n" + "="*70)
print("  🧬 Bloom-Enhanced DNABERT for Sickle Cell Variant Classification")
print("="*70)

print("\nInitializing system...")
print("This may take a minute as models are loaded...\n")

try:
    from app import main
    main()
except KeyboardInterrupt:
    print("\n\nShutting down dashboard...")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error launching dashboard: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check that DNABERT model can be downloaded (requires internet)")
    print("3. Ensure you have ~2GB free memory")
    print("\nFor more help, see README.md or QUICKSTART.md")
    sys.exit(1)
