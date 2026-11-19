"""flightdoc: find out why your drone crashed.

    >>> from flightdoc import analyze
    >>> for d in analyze("00000042.BIN"):
    ...     print(d.verdict)
"""

__version__ = "0.1.0"

from .analysis import Diagnosis, analyze, diagnose  # noqa: E402
from .formats import load  # noqa: E402

__all__ = ["Diagnosis", "analyze", "diagnose", "load", "__version__"]
