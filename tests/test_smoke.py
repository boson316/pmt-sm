"""Smoke tests: import and minimal sanity checks."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_import_pmt_sm():
    import pmt_sm
    assert hasattr(pmt_sm, "__version__")


def test_root_scripts_exist():
    assert (ROOT / "train_pmt_sm.py").exists()
    assert (ROOT / "train_pmt_sm_fast.py").exists()
    assert (ROOT / "app_gradio.py").exists()
