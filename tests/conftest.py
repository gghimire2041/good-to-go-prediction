import sys
from pathlib import Path


# Ensure project root is on sys.path so that `import src.g2g_model...` works in CI and local runs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

