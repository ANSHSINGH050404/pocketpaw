"""Human-friendly error messages for common LLM backend failures.

This module provides utilities to catch and transform cryptic errors into
user-friendly messages for the chat interface.
"""

from dataclasses import dataclass


@dataclass
class BackendError:
    """Represents a user-friendly backend error message."""

    message: str
    suggestion: str
    is_recoverable: bool = True


def classify_error(error: Exception, backend: str) -> BackendError | None:
    """Classify an exception and return a human-friendly error message.

    Args:
        error: The exception to classify
        backend: The backend name (claude_sdk, pocketpaw_native, open_interpreter)

    Returns:
        BackendError with user-friendly message, or None if not a known error type
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Anthropic SDK errors
    if backend in ("claude_sdk", "pocketpaw_native", "anthropic"):
        if _is_anthropic_auth_error(error, error_str, error_type):
            return BackendError(
                message="Can't reach Anthropic API — check your API key in Settings",
                suggestion=(
                    "Open Settings → API Keys in the sidebar and verify your "
                    "ANTHROPIC_API_KEY is correct."
                ),
                is_recoverable=True,
            )
        if _is_anthropic_rate_limit(error, error_str, error_type):
            return BackendError(
                message="Anthropic API rate limit exceeded — please wait a moment",
                suggestion="Wait a minute before sending another message.",
                is_recoverable=True,
            )
        if _is_connection_error(error, error_str, error_type):
            return BackendError(
                message="Can't connect to Anthropic API — check your internet connection",
                suggestion="Verify your internet is working and try again.",
                is_recoverable=True,
            )

    # OpenAI SDK errors
    if backend in ("openai", "open_interpreter"):
        if _is_openai_auth_error(error, error_str, error_type):
            return BackendError(
                message="Can't reach OpenAI API — check your API key in Settings",
                suggestion=(
                    "Open Settings → API Keys in the sidebar and verify your "
                    "OPENAI_API_KEY is correct."
                ),
                is_recoverable=True,
            )
        if _is_openai_rate_limit(error, error_str, error_type):
            return BackendError(
                message="OpenAI API rate limit exceeded — please wait a moment",
                suggestion="Wait a minute before sending another message.",
                is_recoverable=True,
            )

    # Ollama errors
    if backend in ("ollama", "open_interpreter"):
        if _is_ollama_error(error, error_str, error_type):
            return BackendError(
                message="Ollama isn't running — start it with `ollama serve`",
                suggestion=(
                    "Make sure Ollama is installed and running. "
                    "Run 'ollama serve' in a terminal, or download from ollama.ai"
                ),
                is_recoverable=True,
            )

    # Generic connection errors
    if _is_generic_connection_error(error, error_str, error_type):
        return BackendError(
            message="Can't connect to the API — check your internet connection",
            suggestion="Verify your internet is working and try again.",
            is_recoverable=True,
        )

    # Generic timeout errors
    if _is_timeout_error(error, error_str, error_type):
        return BackendError(
            message="Request timed out — the server took too long to respond",
            suggestion=(
                "This can happen if the server is busy or your internet is slow. "
                "Try again in a moment."
            ),
            is_recoverable=True,
        )

    return None


def _is_anthropic_auth_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is an Anthropic authentication error."""
    try:
        from anthropic import AuthenticationError
        if isinstance(error, AuthenticationError):
            return True
    except ImportError:
        pass
    return "authentication" in error_str or "api key" in error_str or "unauthorized" in error_str


def _is_anthropic_rate_limit(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is an Anthropic rate limit error."""
    try:
        from anthropic import RateLimitError

        if isinstance(error, RateLimitError):
            return True
    except ImportError:
        pass
    return (
        "rate_limit" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
    )


def _is_openai_auth_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is an OpenAI authentication error."""
    try:
        from openai import AuthenticationError

        if isinstance(error, AuthenticationError):
            return True
    except ImportError:
        pass
    return (
        "authentication" in error_str
        or "api key" in error_str
        or "unauthorized" in error_str
        or "invalid_api_key" in error_str
    )


def _is_openai_rate_limit(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is an OpenAI rate limit error."""
    try:
        from openai import RateLimitError

        if isinstance(error, RateLimitError):
            return True
    except ImportError:
        pass
    return (
        "rate_limit" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
    )


def _is_ollama_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is an Ollama connection error."""
    return (
        "ollama" in error_str or
        "connection refused" in error_str or
        "connect: cannot connect" in error_str or
        "HTTPConnectionPool" in error_str or
        "Failed to establish a new connection" in error_str or
        error_type in ("ConnectionRefusedError", "ConnectionError", "httpx.ConnectError")
    )


def _is_connection_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is a generic connection error."""
    try:
        from anthropic import APIConnectionError
        if isinstance(error, APIConnectionError):
            return True
    except ImportError:
        pass
    try:
        from openai import APIConnectionError
        if isinstance(error, APIConnectionError):
            return True
    except ImportError:
        pass
    return (
        "connection" in error_str or
        "connection error" in error_str or
        "failed to connect" in error_str or
        "cannot connect" in error_str or
        error_type in ("ConnectionError", "ConnectionRefusedError", "OSError")
    )


def _is_generic_connection_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is a generic network/connection error."""
    return (
        _is_connection_error(error, error_str, error_type) or
        "network" in error_str or
        "socket" in error_str or
        "httpx" in error_str
    )


def _is_timeout_error(error: Exception, error_str: str, error_type: str) -> bool:
    """Check if this is a timeout error."""
    return (
        isinstance(error, TimeoutError) or
        "timeout" in error_str or
        "timed out" in error_str or
        "deadline exceeded" in error_str
    )


def format_error_for_user(error: Exception, backend: str) -> str:
    """Format an exception into a user-friendly message.

    Args:
        error: The exception to format
        backend: The backend name

    Returns:
        A user-friendly error message string
    """
    classified = classify_error(error, backend)

    if classified:
        return f"{classified.message}\n\n{classified.suggestion}"

    # Fallback: sanitize the original error
    sanitized = _sanitize_error_message(str(error))
    return f"An error occurred: {sanitized}"


def _sanitize_error_message(message: str) -> str:
    """Remove sensitive information from error messages."""
    import re

    sanitized = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[API_KEY]", message)
    sanitized = re.sub(
        r"ANTHROPIC_API_KEY=[a-zA-Z0-9-]+", "ANTHROPIC_API_KEY=[REDACTED]", sanitized
    )
    sanitized = re.sub(
        r"OPENAI_API_KEY=[a-zA-Z0-9-]+", "OPENAI_API_KEY=[REDACTED]", sanitized
    )

    if len(sanitized) > 300:
        sanitized = sanitized[:300] + "..."

    return sanitized
