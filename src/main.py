
from pathlib import Path
import sys

# 让 root 目录可 import
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.train_step_cidgmed import main

if __name__ == "__main__":
    main()

