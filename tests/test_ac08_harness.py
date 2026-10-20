"""Tests for AC-08 harness (attribute search precision@10)."""

import pytest
from trace_search.ac08_harness import (
    AC08Harness,
    LabelledRecord,
    AttributeSearchQuery,
)
from trace_search.attributes import PersonAttributes, Color


class TestAC08Harness:
    """Test precision@10 evaluation."""
    
    @pytest.fixture
    def harness(self):
        """Create harness with sample records."""
        h = AC08Harness()
        
        # Add 20 people with various attributes
        records = [
            # Red shirt group (5 people)
            LabelledRecord("p001", PersonAttributes(Color.RED, Color.BLUE, False, 0.9, 0.8)),
            LabelledRecord("p002", PersonAttributes(Color.RED, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("p003", PersonAttributes(Color.RED, Color.BLUE, True, 0.9, 0.8)),
            LabelledRecord("p004", PersonAttributes(Color.RED, Color.GRAY, False, 0.8, 0.7)),
            LabelledRecord("p005", PersonAttributes(Color.RED, Color.BLUE, False, 0.9, 0.9)),
            
            # Blue shirt group (3 people)
            LabelledRecord("p006", PersonAttributes(Color.BLUE, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("p007", PersonAttributes(Color.BLUE, Color.BLUE, True, 0.8, 0.8)),
            LabelledRecord("p008", PersonAttributes(Color.BLUE, Color.GRAY, False, 0.9, 0.7)),
            
            # White shirt group (4 people)
            LabelledRecord("p009", PersonAttributes(Color.WHITE, Color.BLACK, True, 0.9, 0.9)),
            LabelledRecord("p010", PersonAttributes(Color.WHITE, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("p011", PersonAttributes(Color.WHITE, Color.BLUE, False, 0.8, 0.8)),
            LabelledRecord("p012", PersonAttributes(Color.WHITE, Color.BLACK, True, 0.9, 0.9)),
            
            # Other colors
            LabelledRecord("p013", PersonAttributes(Color.GREEN, Color.BROWN, False, 0.9, 0.7)),
            LabelledRecord("p014", PersonAttributes(Color.BLACK, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("p015", PersonAttributes(Color.YELLOW, Color.BLUE, True, 0.8, 0.8)),
            LabelledRecord("p016", PersonAttributes(Color.GRAY, Color.BLACK, False, 0.7, 0.9)),
            LabelledRecord("p017", PersonAttributes(Color.ORANGE, Color.GREEN, False, 0.8, 0.8)),
            LabelledRecord("p018", PersonAttributes(Color.PURPLE, Color.BLACK, True, 0.9, 0.9)),
            LabelledRecord("p019", PersonAttributes(Color.CYAN, Color.BLUE, False, 0.8, 0.8)),
            LabelledRecord("p020", PersonAttributes(Color.PINK, Color.WHITE, False, 0.7, 0.8)),
        ]
        
        for r in records:
            h.add_record(r)
        
        return h
    
    def test_search_red_shirt(self, harness):
        """Search for red shirt."""
        results = harness.search({"upper_color": "red"}, k=10)
        
        # Should return all 5 red-shirt people
        assert len(results) == 5
        assert "p001" in results
        assert "p002" in results
        assert "p003" in results
        assert "p004" in results
        assert "p005" in results
    
    def test_search_with_bag(self, harness):
        """Search for people with bags."""
        results = harness.search({"has_bag": True}, k=10)
        
        # Should find: p003, p007, p009, p012, p015, p018
        assert len(results) == 6
        assert "p003" in results
        assert "p007" in results
        assert "p009" in results
        assert "p012" in results
        assert "p015" in results
        assert "p018" in results
    
    def test_search_combined(self, harness):
        """Search for red shirt AND bag."""
        results = harness.search({"upper_color": "red", "has_bag": True}, k=10)
        
        # Only p003 matches
        assert len(results) == 1
        assert "p003" in results
    
    def test_precision_at_10_perfect(self, harness):
        """Perfect precision@10 when all retrieved are relevant."""
        query = AttributeSearchQuery(
            query_id="q1",
            query={"upper_color": "red"},
            relevant_person_ids=["p001", "p002", "p003", "p004", "p005"],
        )
        
        result = harness.evaluate([query])
        
        # All 5 retrieved are relevant, so precision = 5/5 = 1.0
        assert result["mean_precision_at_10"] == 1.0
        assert result["per_query_results"][0]["precision_at_10"] == 1.0
    
    def test_precision_at_10_partial(self, harness):
        """Partial precision when some retrieved are not relevant."""
        # Query for white shirt, but only mark 2 as relevant
        query = AttributeSearchQuery(
            query_id="q2",
            query={"upper_color": "white"},
            relevant_person_ids=["p009", "p010"],  # Only 2 of 4 marked relevant
        )
        
        result = harness.evaluate([query])
        
        # Retrieved 4, but only 2 marked relevant: precision = 2/4 = 0.5
        assert result["per_query_results"][0]["retrieved_count"] == 4
        assert result["per_query_results"][0]["relevant_in_top10"] == 2
        assert result["per_query_results"][0]["precision_at_10"] == 0.5
    
    def test_multiple_queries(self, harness):
        """Evaluate multiple queries and compute mean."""
        queries = [
            AttributeSearchQuery(
                "q1",
                {"upper_color": "red"},
                ["p001", "p002", "p003", "p004", "p005"],  # All 5 relevant
            ),
            AttributeSearchQuery(
                "q2",
                {"has_bag": True},
                ["p003", "p007", "p009"],  # 3 of 6 relevant
            ),
        ]
        
        result = harness.evaluate(queries)
        
        # q1: 5/5 = 1.0
        # q2: 3/6 = 0.5
        # mean: (1.0 + 0.5) / 2 = 0.75
        assert result["num_queries"] == 2
        assert result["mean_precision_at_10"] == pytest.approx(0.75)
    
    def test_threshold_met(self, harness):
        """AC-08 threshold (0.80) should be reported."""
        # Create high-precision queries
        queries = [
            AttributeSearchQuery("q1", {"upper_color": "red"}, ["p001", "p002", "p003", "p004", "p005"]),
            AttributeSearchQuery("q2", {"upper_color": "blue"}, ["p006", "p007", "p008"]),
        ]
        
        result = harness.evaluate(queries)
        
        # Both should be 1.0, mean = 1.0 >= 0.80
        assert result["threshold_met"] is True
    
    def test_threshold_not_met(self, harness):
        """Low precision should fail AC-08."""
        queries = [
            AttributeSearchQuery("q1", {"upper_color": "white"}, ["p009"]),  # 1/4 = 0.25
        ]
        
        result = harness.evaluate(queries)
        
        assert result["mean_precision_at_10"] < 0.80
        assert result["threshold_met"] is False
    
    def test_json_output(self, harness):
        """JSON output should be valid."""
        import json
        
        queries = [
            AttributeSearchQuery("q1", {"upper_color": "red"}, ["p001", "p002", "p003", "p004", "p005"]),
        ]
        
        json_str = harness.evaluate_to_json(queries)
        result = json.loads(json_str)
        
        assert "mean_precision_at_10" in result
        assert "per_query_results" in result
        assert "num_queries" in result
        assert "threshold_met" in result
