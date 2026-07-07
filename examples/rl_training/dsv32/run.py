import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import adapter
from harness import main

if __name__ == "__main__":
    main(adapter)
