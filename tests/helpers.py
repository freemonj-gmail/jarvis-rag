"""Shared helpers for the Jarvis unit tests.
"""

import contextlib
from unittest.mock import patch


def _patch_answer(name, new=None, **kwargs):
    """Patch ``name`` in every answer module that binds it, with one shared mock.

    The answer package is split across modules that import each other's helpers,
    so a patch on one module would miss calls made from another.
    """
    import importlib
    import pkgutil
    import jarvisrag.answer as package
    from unittest.mock import MagicMock

    modules = [importlib.import_module(f"jarvisrag.answer.{m.name}") for m in pkgutil.iter_modules(package.__path__)]
    holders = [m for m in modules if name in vars(m)]
    original = vars(holders[0])[name]
    holders = [m for m in holders if vars(m)[name] is original]
    replacement = new if new is not None else MagicMock(**kwargs)
    stack = contextlib.ExitStack()
    for module in holders:
        stack.enter_context(patch.object(module, name, replacement))

    @contextlib.contextmanager
    def patched():
        with stack:
            yield replacement

    return patched()
