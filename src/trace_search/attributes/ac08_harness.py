"""
AC-08 harness: Attribute search precision@10 evaluation.

Computes precision@10 for attribute queries from labelled records.
Emits JSON results (not scored in code-only mode).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
from .attributes import PersonAttributes, AttributeQuery


@dataclass
class LabelledRecord:
    """A labelled person with ground-truth attributes."""
    person_id: str
    attributes: PersonAttributes


@dataclass
class AttributeSearchQuery:
    """An attribute search query with ground truth."""
    query_id: str
    query: Dict[str, Any]
    relevant_person_ids: List[str]


class AC08Harness:
    """
    Precision@10 evaluator for attribute queries.
    
    Protocol:
    1. Index a set of labelled person records
    2. Run attribute queries and retrieve top-10 matches
    3. Compute precision@10 for each query
    4. Report mean precision@10 across queries
    """
    
    def __init__(self):
        self.records: List[LabelledRecord] = []
    
    def add_record(self, record: LabelledRecord):
        """Add a labelled person record to the index."""
        self.records.append(record)
    
    def search(self, query: Dict[str, Any], k: int = 10) -> List[str]:
        """
        Search for persons matching the attribute query.
        
        Args:
            query: Attribute query dict (see AttributeQuery.matches)
            k: Number of results to return
        
        Returns:
            List of person_ids ranked by confidence
        """
        matches = []
        
        for record in self.records:
            if AttributeQuery.matches(record.attributes, query):
                # Score by average confidence
                avg_conf = (
                    record.attributes.upper_confidence +
                    record.attributes.lower_confidence
                ) / 2.0
                matches.append((record.person_id, avg_conf))
        
        # Sort by confidence descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return [pid for pid, _ in matches[:k]]
    
    def evaluate(self, queries: List[AttributeSearchQuery]) -> Dict[str, Any]:
        """
        Evaluate precision@10 across multiple queries.
        
        Args:
            queries: List of queries with ground-truth relevant IDs
        
        Returns:
            Evaluation results dict with:
                - per_query_results: List of per-query precision@10
                - mean_precision_at_10: Mean across queries
                - num_queries: Total queries evaluated
        """
        results = []
        
        for q in queries:
            retrieved = self.search(q.query, k=10)
            relevant_set = set(q.relevant_person_ids)
            
            # Precision@10: fraction of top-10 that are relevant
            relevant_in_top10 = sum(1 for pid in retrieved if pid in relevant_set)
            precision_at_10 = relevant_in_top10 / min(10, len(retrieved)) if retrieved else 0.0
            
            results.append({
                "query_id": q.query_id,
                "query": q.query,
                "precision_at_10": precision_at_10,
                "retrieved_count": len(retrieved),
                "relevant_in_top10": relevant_in_top10,
            })
        
        mean_p10 = sum(r["precision_at_10"] for r in results) / len(results) if results else 0.0
        
        return {
            "per_query_results": results,
            "mean_precision_at_10": mean_p10,
            "num_queries": len(queries),
            "threshold_met": mean_p10 >= 0.80,  # AC-08 threshold
        }
    
    def evaluate_to_json(self, queries: List[AttributeSearchQuery]) -> str:
        """Run evaluation and return JSON string."""
        results = self.evaluate(queries)
        return json.dumps(results, indent=2)
