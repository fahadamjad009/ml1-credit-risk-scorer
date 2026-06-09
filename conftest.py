import sys
from pathlib import Path

# Add project root to sys.path so `from src.x import y` works in tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
