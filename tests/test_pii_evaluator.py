"""Tests for the PII Evaluator."""

import pytest
from unittest.mock import MagicMock
from evalview.evaluators.pii_evaluator import PIIEvaluator

@pytest.mark.asyncio
async def test_pii_evaluator_passes_clean_text():
    """Test that a clean response passes the PII check."""
    evaluator = PIIEvaluator()
    
    # Mock a clean execution trace
    mock_trace = MagicMock()
    mock_trace.final_output = "Hello world, the weather is nice today."
    
    # Mock an empty test case (unused by this evaluator)
    mock_test_case = MagicMock()

    # Execute the evaluator
    result = await evaluator.evaluate(test_case=mock_test_case, trace=mock_trace)

    # Verify the results
    assert result.passed is True
    assert result.has_pii is False
    assert len(result.types_detected) == 0

@pytest.mark.asyncio
async def test_pii_evaluator_fails_with_pii():
    """Test that a response with email and phone fails the PII check."""
    evaluator = PIIEvaluator()
    
    # Mock a trace containing PII violations
    mock_trace = MagicMock()
    mock_trace.final_output = "You can reach me at john.doe@example.com or call me at (123) 456-7890."
    
    mock_test_case = MagicMock()

    # Execute the evaluator
    result = await evaluator.evaluate(test_case=mock_test_case, trace=mock_trace)

    # Verify that it correctly fails and identifies the PII types
    assert result.passed is False
    assert result.has_pii is True
    assert "email" in result.types_detected
    assert "phone" in result.types_detected

@pytest.mark.asyncio
async def test_pii_evaluator_empty_output():
    """Test that an empty output is handled gracefully and passes."""
    evaluator = PIIEvaluator()
    
    # Mock a trace with empty output
    mock_trace = MagicMock()
    mock_trace.final_output = ""
    
    mock_test_case = MagicMock()

    result = await evaluator.evaluate(test_case=mock_test_case, trace=mock_trace)

    assert result.passed is True
    assert result.has_pii is False