"""Deterministic mock configuration for the gym agent.

This module provides predictable, repeatable responses for testing.
No randomness, no API calls - just consistent behavior for learning evals.

Environment variables:
    EVALVIEW_MOCK_MODE: "always" | "never" | "auto" (default: "always")
    EVALVIEW_MOCK_SEED: int (default: 42) - for any future randomization
"""

import os
from typing import Any, Dict

# Mock mode configuration
MOCK_MODE = os.getenv("EVALVIEW_MOCK_MODE", "always")
MOCK_SEED = int(os.getenv("EVALVIEW_MOCK_SEED", "42"))

# Deterministic response registry
# These responses are designed to be predictable and testable
MOCK_RESPONSES: Dict[str, Dict[str, Any]] = {
    "search_kb": {
        "refund": {
            "found": True,
            "article_id": "KB-001",
            "title": "Refund Policy",
            "content": "Refunds are available within 30 days of purchase. "
            "To request a refund, provide your order number and reason.",
            "confidence": 0.95,
        },
        "shipping": {
            "found": True,
            "article_id": "KB-002",
            "title": "Shipping Information",
            "content": "Standard shipping takes 5-7 business days. "
            "Express shipping (2-3 days) is available for $9.99.",
            "confidence": 0.92,
        },
        "password": {
            "found": True,
            "article_id": "KB-003",
            "title": "Password Reset",
            "content": "To reset your password, click 'Forgot Password' on the login page. "
            "A reset link will be sent to your registered email.",
            "confidence": 0.98,
        },
        "cancel": {
            "found": True,
            "article_id": "KB-004",
            "title": "Order Cancellation",
            "content": "Orders can be cancelled within 1 hour of placement. "
            "After that, please wait for delivery and request a refund.",
            "confidence": 0.90,
        },
        "default": {
            "found": False,
            "article_id": None,
            "title": None,
            "content": "No relevant article found. Consider creating a support ticket.",
            "confidence": 0.0,
        },
    },
    "create_ticket": {
        "default": {
            "ticket_id": "TKT-{priority}-001",
            "status": "created",
            "estimated_response": "24 hours",
        },
    },
    "send_reply": {
        "default": {
            "sent": True,
            "message_id": "MSG-001",
            "timestamp": "2024-01-15T10:30:00Z",
        },
    },
    "check_order": {
        "12345": {
            "order_id": "12345",
            "status": "shipped",
            "tracking": "1Z999AA10123456784",
            "estimated_delivery": "2024-01-18",
        },
        "99999": {
            "order_id": "99999",
            "status": "processing",
            "tracking": None,
            "estimated_delivery": "2024-01-20",
        },
        "default": {
            "order_id": None,
            "status": "not_found",
            "error": "Order not found. Please check the order number.",
        },
    },
}


def get_mock_response(tool: str, query: str, **kwargs) -> Any:
    """Get deterministic mock response based on tool and query.

    Args:
        tool: The tool name (e.g., "search_kb", "create_ticket")
        query: The query or key to look up
        **kwargs: Additional context (e.g., priority for tickets)

    Returns:
        Deterministic response based on query content
    """
    responses = MOCK_RESPONSES.get(tool, {})

    # Try to match query content to a known response
    query_lower = query.lower() if isinstance(query, str) else str(query).lower()

    for key, value in responses.items():
        if key != "default" and key in query_lower:
            # Handle template substitution
            if isinstance(value, dict):
                result = value.copy()
                if "ticket_id" in result and "{priority}" in str(result["ticket_id"]):
                    priority = kwargs.get("priority", "medium")
                    result["ticket_id"] = result["ticket_id"].format(priority=priority.upper())
                return result
            return value

    # Return default response
    default = responses.get("default", {"error": f"No mock configured for {tool}"})
    if isinstance(default, dict):
        result = default.copy()
        if "ticket_id" in result and "{priority}" in str(result["ticket_id"]):
            priority = kwargs.get("priority", "medium")
            result["ticket_id"] = result["ticket_id"].format(priority=priority.upper())
        return result
    return default


def is_mock_mode() -> bool:
    """Check if we should use mock responses."""
    return MOCK_MODE == "always" or (MOCK_MODE == "auto" and not _has_real_api_keys())


def _has_real_api_keys() -> bool:
    """Check if real API keys are configured."""
    return bool(os.getenv("OPENAI_API_KEY"))
