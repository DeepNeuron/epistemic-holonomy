import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "replication"))


@pytest.fixture(scope="session")
def certificate():
    """The repository's original certificate module, imported for cross-checks only."""
    spec = importlib.util.spec_from_file_location(
        "epistemic_holonomy_certificate", ROOT / "epistemic_holonomy_certificate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def results():
    import json
    out = {}
    for p in (ROOT / "replication" / "results").glob("*.json"):
        out[p.stem] = json.loads(p.read_text())
    return out
