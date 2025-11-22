"""JSON reporter for evaluation results."""

import json
from pathlib import Path
from typing import List, Union, Dict, Any
from evalview.core.types import EvaluationResult


class JSONReporter:
    """Generates JSON reports from evaluation results."""

    @staticmethod
    def save(results: List[EvaluationResult], output_path: Union[str, Path]) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: List of evaluation results
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        results_dict = [result.model_dump(mode="json") for result in results]

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

    @staticmethod
    def load(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load evaluation results from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of result dictionaries
        """
        with open(input_path, "r") as f:
            return json.load(f)
