project = "vector-structure"
author = "Pierre Ballif <pierre@ballif.eu>"

# Find package even if it is not installed
# fmt: off
import sys  # noqa: E402, I001
from pathlib import Path  # noqa: E402, I001
sys.path.append(str(Path(__file__).parent.parent.parent))
# fmt: on

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
