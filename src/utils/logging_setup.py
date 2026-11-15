"""Project-wide logging configuration.

Reads the ``logging`` section of ``config.yaml`` so the level, message format
and log file live in one place. Call :func:`configure_logging` once at the start
of an entry point; every other module then obtains its logger with
:func:`get_logger`.

Typical use::

    from src.utils.logging_setup import get_logger

    log = get_logger(__name__)
    log.info("starting")
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

from src.utils.config import ConfigError, get_logging_config, resolve_path

__all__ = ["configure_logging", "get_logger"]

#: Fallback format used when the config file cannot be read.
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_LEVEL = "INFO"

_configured = False


def _resolve_level(value: Optional[str]) -> int:
    """Translate a level name such as ``"INFO"`` into a ``logging`` constant."""
    if not value:
        return logging.INFO
    level = getattr(logging, str(value).upper(), None)
    return level if isinstance(level, int) else logging.INFO


def configure_logging(
    level: Optional[str] = None,
    *,
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger from ``config.yaml``.

    Args:
        level: Overrides ``logging.level`` from the config file.
        log_file: Overrides ``logging.file`` from the config file (project-root
            relative). Pass ``""`` to disable the file handler.
        force: Re-apply the handlers even if logging was already configured.

    Returns:
        The configured root logger.
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    try:
        config = get_logging_config()
    except ConfigError:
        config = {}

    resolved_format = config.get("format") or _DEFAULT_FORMAT
    resolved_level = _resolve_level(level or config.get("level") or _DEFAULT_LEVEL)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    target = config.get("file") if log_file is None else log_file
    if target:
        path = resolve_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(resolved_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(resolved_level)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
