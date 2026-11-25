"""Keep private text out of evaluation output."""

from __future__ import annotations

import contextlib
import logging
import os


@contextlib.contextmanager
def private_runtime_output(verbose: bool = False):
    """Hide pipeline logs and printed answers, which can quote private
    documents, unless ``verbose`` asks to see them."""
    if verbose:
        yield
        return
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            with contextlib.redirect_stdout(null_stream), contextlib.redirect_stderr(null_stream):
                yield
    finally:
        logging.disable(previous_disable)
