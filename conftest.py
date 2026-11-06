"""pytest config: make the smart_table modules importable from tests/."""

import os
import sys

# Add the smart_table directory (parent of tests/) to the import path so the
# tests can do "import config", "import seated_logic", etc.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
