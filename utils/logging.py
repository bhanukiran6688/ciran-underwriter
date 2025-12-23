"""
Minimal structured logging helpers for the POC.

Usage:
    from utils.logging import get_logger
    log = get_logger(__name__)
    log.info("message", extra={"ctx": {"request_id": "...", "node": "..."}})
"""

import json
import logging
import sys
from typing import Any, Dict


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Accept `extra={"ctx": {...}}`
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            payload["ctx"] = ctx
        return json.dumps(payload, ensure_ascii=False)


def _configure_root(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    # Clear existing handlers only once.
    if getattr(root, "_ciran_configured", False):
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    setattr(root, "_ciran_configured", True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a module logger with JSON formatting attached to the root once.
    """
    _configure_root()
    return logging.getLogger(name)
