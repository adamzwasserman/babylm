"""Shared pytest setup for the babylm test suite.

The scripts under scripts/ pull in heavy data-science dependencies (tqdm,
datasets, torch, transformers, ...) at import time. Most of those are not
needed to exercise the small pure-Python helpers we test here. To keep the
test suite installable and fast, we stub the heaviest optional imports when
they are not present in the environment.

If a real dependency is needed by a particular test, that test should
import it directly and skip via pytest.importorskip.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def _stub_tqdm() -> None:
    if importlib.util.find_spec("tqdm") is not None:
        return
    module = types.ModuleType("tqdm")

    def passthrough(iterable=None, *args, **kwargs):
        if iterable is None:
            return _NoOp()
        return iterable

    class _NoOp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

    module.tqdm = passthrough
    sys.modules["tqdm"] = module


_stub_tqdm()
