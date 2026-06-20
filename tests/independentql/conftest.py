"""Make the `independent_ql` package importable in tests.

The Independent-QL folder is hyphenated (not a valid Python package name) and its
modules import the package bare (``from independent_ql...``), the same way the
entrypoint works when run as ``python Independent-QL/main.py``. Putting the
folder on sys.path here lets the tests import it from the repo root.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PKG_DIR = os.path.join(_REPO_ROOT, "Independent-QL")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
