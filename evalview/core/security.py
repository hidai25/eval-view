"""Security utilities for EvalView.

This module provides security-related utilities including:
- URL validation to prevent SSRF attacks
- Input sanitization for LLM prompts
"""

import ipaddress
import re
import socket
from typing import Optional, Set
from urllib.parse import urlparse


class SSRFProtectionError(Exception):
    """Raised when a URL fails SSRF protection checks."""

    pass


# Private/internal IP ranges that should be blocked by default
BLOCKED_IP_RANGES = [
    # Loopback
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    # Private networks (RFC 1918)
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    # Link-local
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
    # Cloud metadata endpoints
    ipaddress.ip_network("169.254.169.254/32"),  # AWS/GCP/Azure metadata
    # Localhost IPv6
    ipaddress.ip_network("::ffff:127.0.0.0/104"),
    # Private IPv6
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fd00::/8"),
]

# Hostnames that should be blocked
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.goog",
    "kubernetes.default.svc",
    "kubernetes.default",
}

# Schemes that are allowed
ALLOWED_SCHEMES = {"http", "https"}


def is_ip_blocked(ip_str: str) -> bool:
    """
    Check if an IP address is in a blocked range.

    Args:
        ip_str: IP address string

    Returns:
        True if the IP is blocked, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        for network in BLOCKED_IP_RANGES:
            if ip in network:
                return True
        return False
    except ValueError:
        # Not a valid IP address
        return False


def resolve_hostname(hostname: str) -> Optional[str]:
    """
    Resolve a hostname to its IP address.

    Args:
        hostname: Hostname to resolve

    Returns:
        IP address string or None if resolution fails
    """
    try:
        # Get all IP addresses for the hostname
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        if infos:
            # Return the first IP address
            return infos[0][4][0]
    except (socket.gaierror, socket.herror, OSError):
        pass
    return None


def validate_url(
    url: str,
    allow_private: bool = False,
    allowed_hosts: Optional[Set[str]] = None,
    blocked_hosts: Optional[Set[str]] = None,
    resolve_dns: bool = True,
) -> str:
    """
    Validate a URL for SSRF protection.

    This function checks that a URL:
    1. Uses an allowed scheme (http/https)
    2. Does not target private/internal IP ranges
    3. Does not target blocked hostnames
    4. Resolves to a non-private IP (optional DNS check)

    Args:
        url: URL to validate
        allow_private: If True, allow private/internal IPs (for local development)
        allowed_hosts: Optional set of explicitly allowed hostnames
        blocked_hosts: Optional additional hostnames to block
        resolve_dns: If True, resolve hostname and check resulting IP

    Returns:
        The validated URL (unchanged if valid)

    Raises:
        SSRFProtectionError: If the URL fails validation
    """
    if not url:
        raise SSRFProtectionError("URL cannot be empty")

    # Parse the URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFProtectionError(f"Invalid URL format: {e}")

    # Check scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SSRFProtectionError(
            f"URL scheme '{parsed.scheme}' is not allowed. Use http or https."
        )

    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        raise SSRFProtectionError("URL must contain a hostname")

    hostname_lower = hostname.lower()

    # Check if explicitly allowed
    if allowed_hosts and hostname_lower in {h.lower() for h in allowed_hosts}:
        return url

    # Skip all private/localhost checks if allowed (for local development)
    if allow_private:
        return url

    # Check blocked hostnames
    all_blocked = BLOCKED_HOSTNAMES.copy()
    if blocked_hosts:
        all_blocked.update(h.lower() for h in blocked_hosts)

    if hostname_lower in all_blocked:
        raise SSRFProtectionError(
            f"Hostname '{hostname}' is blocked for security reasons"
        )

    # Check if hostname is a direct IP address
    if is_ip_blocked(hostname):
        raise SSRFProtectionError(
            f"IP address '{hostname}' is in a blocked range (private/internal network)"
        )

    # Resolve DNS and check the resulting IP
    if resolve_dns:
        resolved_ip = resolve_hostname(hostname)
        if resolved_ip and is_ip_blocked(resolved_ip):
            raise SSRFProtectionError(
                f"Hostname '{hostname}' resolves to blocked IP '{resolved_ip}' "
                "(private/internal network)"
            )

    return url


def sanitize_for_llm(
    text: str,
    max_length: int = 10000,
    escape_delimiters: bool = True,
) -> str:
    """
    Sanitize text before including it in LLM prompts to mitigate prompt injection.

    This function:
    1. Truncates text to a maximum length
    2. Optionally escapes common prompt delimiters
    3. Removes null bytes and control characters

    Args:
        text: Text to sanitize
        max_length: Maximum allowed length (default 10000 chars)
        escape_delimiters: If True, escape common prompt delimiters

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes and most control characters (keep newlines and tabs)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "\n[... OUTPUT TRUNCATED ...]"

    # Escape common prompt injection delimiters if requested
    if escape_delimiters:
        # Escape triple backticks (common code block delimiter)
        sanitized = sanitized.replace("```", "` ` `")
        # Escape XML-like tags that might be interpreted as instructions
        sanitized = re.sub(r"<(/?)(system|user|assistant|instruction)>", r"[\1\2]", sanitized, flags=re.IGNORECASE)
        # Escape common prompt boundary markers
        sanitized = sanitized.replace("###", "# # #")
        sanitized = sanitized.replace("---", "- - -")

    return sanitized


def create_safe_llm_boundary(identifier: str) -> tuple:
    """
    Create a unique boundary marker for safely delimiting untrusted content in LLM prompts.

    Args:
        identifier: A unique identifier for this boundary

    Returns:
        Tuple of (start_marker, end_marker)
    """
    import hashlib
    import time

    # Create a unique hash-based boundary
    unique = hashlib.sha256(f"{identifier}-{time.time()}".encode()).hexdigest()[:16]
    start = f"<<<UNTRUSTED_CONTENT_{unique}>>>"
    end = f"<<<END_UNTRUSTED_CONTENT_{unique}>>>"
    return start, end
