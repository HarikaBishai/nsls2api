"""Push logs to Loki with loki-logger-handler, as in the NSLS-II Loki how-to.

``uvicorn_log_config.yml`` attaches the handler made by :func:`loki_handler` to the
uvicorn loggers, next to the console handlers. Logs are pushed only when ``LOKI_URL``
and ``LOKI_PUSH_SECRET`` are set (the ``loki_*`` fields of ``Settings`` in ``config.py``);
otherwise the handler does nothing.

API keys must not reach the shared Loki tenant, so the handler formats records with
:class:`RedactingLokiFormatter` and the log configuration gives it the same
``RedactApiKeyFilter`` as the console handlers.
"""

import functools
import logging
import socket
import sys
import threading
import traceback
from typing import Any

from loki_logger_handler.formatters.logger_formatter import LoggerFormatter
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler
from pydantic import ValidationError

from nsls2api.infrastructure.config import get_settings
from nsls2api.infrastructure.logging import RedactApiKeyFilter, redact_api_keys

# (connect, read) seconds for each Loki push.
_LOKI_TIMEOUT = (5.0, 10.0)

# Record attributes meant for the console, which the library would copy into every
# line: asctime, left by the console formatters that run first, and the ANSI-coloured
# copy of the message that uvicorn adds to some records.
_CONSOLE_ONLY_FIELDS = ("asctime", "color_message")


def _without_traceback(record: logging.LogRecord) -> bool:
    # The library logs each failed push with its full chained traceback and a
    # multi-line message; one line says what failed.
    record.msg = " ".join(record.getMessage().split())
    record.args = None
    record.exc_info = None
    record.exc_text = None
    return True


class SerializedLokiHandler(LokiLoggerHandler):
    """LokiLoggerHandler whose queue drains run one at a time.

    The library drains with ``while not buffer.empty(): buffer.get()``. When the
    sender thread and the send at exit both see the last record, one takes it and
    the other blocks in ``get()`` forever, so shutdown hangs.

    Overrides the private ``_send()`` of loki-logger-handler 1.1.2 (pinned); recheck on upgrade.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Before super().__init__(), which starts the sender thread.
        self._send_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _send(self) -> None:
        with self._send_lock:
            super()._send()


def _redacted_json_value(value: Any) -> Any:
    """Return ``value`` as JSON-serializable data, with API keys redacted from all text.

    Goes into dicts, lists and tuples. Anything else that is not a JSON scalar becomes
    its ``str()``, because the library serializes a whole batch at once and one value
    it cannot serialize loses every record in that batch.
    """
    if isinstance(value, str):
        return redact_api_keys(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {
            redact_api_keys(str(key)): _redacted_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redacted_json_value(item) for item in value]
    return redact_api_keys(str(value))


class RedactingLokiFormatter(LoggerFormatter):
    """The library's formatter, with API keys redacted and tracebacks at every level.

    The library builds the traceback from ``exc_info``, which ``RedactApiKeyFilter``
    does not change, and copies extra record attributes as they are. Here the
    traceback is the redacted ``exc_text`` the filter left on the record, and every
    field is made JSON-serializable and redacted again, nested values included, before
    the record leaves the process. The library adds a traceback only to records whose
    level starts with "ER"; here any record with an exception gets one.
    """

    def format(
        self, record: logging.LogRecord
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        formatted, loki_metadata = super().format(record)
        for key in _CONSOLE_ONLY_FIELDS:
            formatted.pop(key, None)
        if record.exc_info or record.exc_text:
            formatted["stacktrace"] = record.exc_text or "".join(
                traceback.format_exception(*record.exc_info)
            )
        return _redacted_json_value(formatted), loki_metadata


def loki_handler() -> logging.Handler:
    """Handler factory for the log configuration (``"()"`` in uvicorn_log_config.yml).

    Returns a handler that does nothing unless ``LOKI_URL`` and ``LOKI_PUSH_SECRET``
    are both set, or when the settings are invalid: the application then fails to
    start with the validation error, which is clearer than a log configuration error.
    """
    try:
        # Uncached: this runs while logging is being configured, before the app starts.
        settings = get_settings.__wrapped__()
    except ValidationError:
        return logging.NullHandler()
    secret = settings.loki_push_secret.get_secret_value()
    if not (settings.loki_url and secret):
        return logging.NullHandler()

    labels = {
        "job": settings.loki_job,
        # Same value as job, so apps sharing the tenant can all be found by service_name.
        "service_name": settings.loki_job,
        "environment": settings.loki_environment,
        "host": settings.loki_host or socket.gethostname(),
    }
    handler = SerializedLokiHandler(
        url=settings.loki_url,
        labels={k: v for k, v in labels.items() if v},
        label_keys=["level"],
        additional_headers={"loki-push-secret": secret},
        default_formatter=RedactingLokiFormatter(),
        enable_self_errors=True,
    )
    # Report failed pushes on stderr (the journal), redacted and on one line each,
    # without sending them back to Loki.
    console = logging.StreamHandler(sys.stderr)
    console.addFilter(_without_traceback)
    console.addFilter(RedactApiKeyFilter())
    handler.debug_logger.propagate = False
    handler.debug_logger.handlers.clear()
    handler.debug_logger.addHandler(console)
    # The library posts without a network timeout, so a Loki that accepts the
    # connection but never answers would block its sender, and its send at exit, forever.
    session = handler.request.session
    session.request = functools.partial(session.request, timeout=_LOKI_TIMEOUT)
    return handler
