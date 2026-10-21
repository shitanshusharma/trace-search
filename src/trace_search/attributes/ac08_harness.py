"""
AC-08 harness: Attribute search precision@10 evaluation.

Computes precision@10 for attribute queries from labelled records.
Emits JSON results (DEFERRED in code-only mode, not scored until FX-HOLD).
"""

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from .attributes import AttributeQuery, PersonAttributes


@dataclass
class LabelledRecord:
    """A track with predicted attributes and ground-truth relevance."""

    track_id: str
    attributes: PersonAttributes  # Predicted by AttributeExtractor


@dataclass
class AttributeSearchQuery:
    """An attribute search query with ground truth."""

    query_id: str
    query: dict[str, Any]
    relevant_track_ids: list[str]


class _PerQueryResult(TypedDict):
    """One precision@10 row. Typed so the mean sums floats."""

    query_id: str
    query: dict[str, Any]
    precision_at_10: float
    retrieved_count: int
    relevant_in_top10: int


class AC08Harness:
    """
    Precision@10 evaluator for attribute queries (AC-08, DEFERRED).

    Protocol:
    1. Index a set of track records with PREDICTED attributes
    2. Run attribute queries and retrieve top-10 matches
    3. Compute precision@10 for each query (relevant in top-10 / 10)
    4. Report mean precision@10 across ≥10 queries with fixture_id

    DEFERRED: Not scored in code-only mode. Requires FX-HOLD with ≥10 queries.
    """

    def __init__(self, fixture_id: str):
        """
        Initialize harness.

        Args:
            fixture_id: Fixture identifier (e.g., 'FX-HOLD-001' or 'FX-CODE-001')
        """
        if not fixture_id:
            raise ValueError("fixture_id is required")
        self.fixture_id = fixture_id
        self.records: list[LabelledRecord] = []

    def add_record(self, record: LabelledRecord) -> None:
        """Add a track record with predicted attributes to the index."""
        self.records.append(record)

    def search(self, query: dict[str, Any], k: int = 10) -> list[str]:
        """
        Search for tracks matching the attribute query.

        Args:
            query: Attribute query dict (see AttributeQuery.matches)
            k: Number of results to return

        Returns:
            List of track_ids ranked by predicted attribute confidence (descending)
        """
        matches = []

        for record in self.records:
            if AttributeQuery.matches(record.attributes, query):
                # Rank by average predicted confidence
                avg_conf = (
                    record.attributes.upper_confidence + record.attributes.lower_confidence
                ) / 2.0
                matches.append((record.track_id, avg_conf))

        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return [tid for tid, _ in matches[:k]]

    def evaluate(self, queries: list[AttributeSearchQuery]) -> dict[str, Any]:
        """
        Evaluate precision@10 across multiple queries.

        Args:
            queries: List of queries with ground-truth relevant track IDs (≥10 required)

        Returns:
            Evaluation results dict with:
                - fixture_id: Fixture identifier
                - status: "DEFERRED" (not scored in code-only mode)
                - per_query_results: List of per-query precision@10
                - mean_precision_at_10: Mean across queries
                - num_queries: Total queries evaluated
                - ac08_threshold: 0.80 (for reference, not enforced)

        Raises:
            ValueError: If fewer than 10 queries provided
        """
        if len(queries) < 10:
            raise ValueError(f"AC-08 requires ≥10 queries, got {len(queries)}")

        results: list[_PerQueryResult] = []

        for q in queries:
            retrieved = self.search(q.query, k=10)
            relevant_set = set(q.relevant_track_ids)

            # Precision@10: relevant in top-10 divided by 10 (not by retrieved count)
            relevant_in_top10 = sum(1 for tid in retrieved if tid in relevant_set)
            precision_at_10 = relevant_in_top10 / 10.0

            results.append(
                {
                    "query_id": q.query_id,
                    "query": q.query,
                    "precision_at_10": precision_at_10,
                    "retrieved_count": len(retrieved),
                    "relevant_in_top10": relevant_in_top10,
                }
            )

        mean_p10 = sum(r["precision_at_10"] for r in results) / len(results) if results else 0.0

        return {
            "fixture_id": self.fixture_id,
            "status": "DEFERRED",
            "per_query_results": results,
            "mean_precision_at_10": mean_p10,
            "num_queries": len(queries),
            "ac08_threshold": 0.80,  # Reference only; not enforced in code-only mode
        }

    def evaluate_to_json(self, queries: list[AttributeSearchQuery]) -> str:
        """Run evaluation and return JSON string."""
        results = self.evaluate(queries)
        return json.dumps(results, indent=2)
