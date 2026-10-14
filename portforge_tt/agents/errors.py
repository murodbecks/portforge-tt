"""Shared provider-error taxonomy for every agent CLI backend.

Derived from the ``chia.models.opencode`` / ``chia.models.claude`` exception
hierarchies (Copyright (c) 2026 The Regents of the University of California;
BSD-3-Clause, see LICENSE), unified into one set so the controller classifies
failures identically no matter which CLI a role runs on. ``node_id`` was a Ray
concept and is gone; ``reset_time`` on rate limits is preserved because the
controller honours a provider-reported usage reset.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional


class ProviderError(Exception):
    """Base for all agent-CLI provider errors."""

    error_type = "provider_error"

    def __init__(self, exit_code: int = -1, raw_message: str = ""):
        self.exit_code = exit_code
        self.raw_message = raw_message
        super().__init__(f"{self.error_type}: {raw_message[:200]}")


class RateLimitError(ProviderError):
    """The provider reported a usage/rate limit (may carry a reset time)."""

    error_type = "rate_limit"

    def __init__(self, reset_time: Optional[datetime] = None,
                 raw_message: str = "", exit_code: int = -1):
        self.reset_time = reset_time
        super().__init__(exit_code, raw_message)


class AuthenticationError(ProviderError):
    """The CLI has no or invalid credentials for the selected provider."""

    error_type = "authentication_failed"


class BillingError(ProviderError):
    """The provider account has a billing/payment problem."""

    error_type = "billing_error"


class InvalidRequestError(ProviderError):
    """Malformed request: bad model string, invalid config, unknown agent…"""

    error_type = "invalid_request"


class ServerError(ProviderError):
    """Transient provider/server-side failure (5xx, overloaded, connection)."""

    error_type = "server_error"

    def __init__(self, exit_code: int = -1, raw_message: str = "",
                 retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(exit_code, raw_message)


class MaxOutputTokensError(ProviderError):
    """The response was truncated at the output token limit."""

    error_type = "max_output_tokens"

    def __init__(self, exit_code: int = -1, raw_message: str = "",
                 partial_text: str = ""):
        self.partial_text = partial_text
        super().__init__(exit_code, raw_message)


class UnknownProviderError(ProviderError):
    """Unclassified CLI error; carries stderr for honest reporting."""

    error_type = "unknown"

    def __init__(self, exit_code: int = -1, raw_message: str = "", stderr: str = ""):
        self.stderr = stderr
        super().__init__(exit_code, raw_message)


# Back-compat aliases used by the ported opencode backend and older callers.
OpenCodeError = ProviderError
UnknownOpenCodeError = UnknownProviderError
