"""Tests for AC-08 harness (attribute search precision@10)."""

import json

import pytest

from trace_search.attributes import (
    AC08Harness,
    AttributeSearchQuery,
    Color,
    LabelledRecord,
    PersonAttributes,
)


class TestAC08Harness:
    """Test precision@10 evaluation (DEFERRED in code-only mode)."""

    @pytest.fixture
    def harness_with_records(self):
        """Harness with 20 sample tracks."""
        h = AC08Harness(fixture_id="FX-CODE-001")

        records = [
            # Red shirt group (5 tracks)
            LabelledRecord("t001", PersonAttributes(Color.RED, Color.BLUE, False, 0.9, 0.8)),
            LabelledRecord("t002", PersonAttributes(Color.RED, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("t003", PersonAttributes(Color.RED, Color.BLUE, True, 0.9, 0.8)),
            LabelledRecord("t004", PersonAttributes(Color.RED, Color.GRAY, False, 0.8, 0.7)),
            LabelledRecord("t005", PersonAttributes(Color.RED, Color.BLUE, False, 0.9, 0.9)),
            # Blue shirt group (3 tracks)
            LabelledRecord("t006", PersonAttributes(Color.BLUE, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("t007", PersonAttributes(Color.BLUE, Color.BLUE, True, 0.8, 0.8)),
            LabelledRecord("t008", PersonAttributes(Color.BLUE, Color.GRAY, False, 0.9, 0.7)),
            # White shirt group (4 tracks)
            LabelledRecord("t009", PersonAttributes(Color.WHITE, Color.BLACK, True, 0.9, 0.9)),
            LabelledRecord("t010", PersonAttributes(Color.WHITE, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("t011", PersonAttributes(Color.WHITE, Color.BLUE, False, 0.8, 0.8)),
            LabelledRecord("t012", PersonAttributes(Color.WHITE, Color.BLACK, True, 0.9, 0.9)),
            # Other colors
            LabelledRecord("t013", PersonAttributes(Color.GREEN, Color.BROWN, False, 0.9, 0.7)),
            LabelledRecord("t014", PersonAttributes(Color.BLACK, Color.BLACK, False, 0.9, 0.9)),
            LabelledRecord("t015", PersonAttributes(Color.YELLOW, Color.BLUE, True, 0.8, 0.8)),
            LabelledRecord("t016", PersonAttributes(Color.GRAY, Color.BLACK, False, 0.7, 0.9)),
            LabelledRecord("t017", PersonAttributes(Color.ORANGE, Color.GREEN, False, 0.8, 0.8)),
            LabelledRecord("t018", PersonAttributes(Color.PURPLE, Color.BLACK, True, 0.9, 0.9)),
            LabelledRecord("t019", PersonAttributes(Color.CYAN, Color.BLUE, False, 0.8, 0.8)),
            LabelledRecord("t020", PersonAttributes(Color.PINK, Color.WHITE, False, 0.7, 0.8)),
        ]

        for r in records:
            h.add_record(r)

        return h

    def test_search_red_shirt(self, harness_with_records):
        """Search returns red-shirt tracks ranked by confidence."""
        results = harness_with_records.search({"upper_color": "red"}, k=10)
        assert len(results) == 5
        assert all(tid in ["t001", "t002", "t003", "t004", "t005"] for tid in results)

    def test_search_with_bag(self, harness_with_records):
        """Search for tracks with bags."""
        results = harness_with_records.search({"bag_on_track": True}, k=10)
        assert len(results) == 6
        assert "t003" in results  # red + bag
        assert "t007" in results  # blue + bag

    def test_precision_at_10_perfect(self, harness_with_records):
        """Perfect P@10 when all 5 retrieved are relevant."""
        queries = [
            AttributeSearchQuery(
                "q1",
                {"upper_color": "red"},
                ["t001", "t002", "t003", "t004", "t005"],
            ),
        ] * 10  # Duplicate to meet 10-query floor

        result = harness_with_records.evaluate(queries)

        # All 5 retrieved, all relevant: P@10 = 5/10 = 0.5
        assert result["per_query_results"][0]["precision_at_10"] == 0.5
        assert result["status"] == "DEFERRED"
        assert result["fixture_id"] == "FX-CODE-001"

    def test_precision_at_10_partial(self, harness_with_records):
        """Partial P@10: 3 relevant out of 5 retrieved."""
        queries = [
            AttributeSearchQuery(
                "q1",
                {"upper_color": "white"},
                ["t009", "t010", "t011"],  # Only 3 of 4 marked relevant
            ),
        ] * 10

        result = harness_with_records.evaluate(queries)

        # Retrieved 4, relevant 3: P@10 = 3/10 = 0.3
        assert result["per_query_results"][0]["retrieved_count"] == 4
        assert result["per_query_results"][0]["relevant_in_top10"] == 3
        assert result["per_query_results"][0]["precision_at_10"] == 0.3

    def test_precision_at_10_three_of_five(self, harness_with_records):
        """3 relevant results out of 5 retrieved gives P@10 = 0.3."""
        queries = [
            AttributeSearchQuery(
                "q_partial",
                {"upper_color": "red"},
                ["t001", "t002", "t003"],  # Only 3 of 5 red shirts relevant
            ),
        ] * 10

        result = harness_with_records.evaluate(queries)

        # Retrieved 5, relevant 3: P@10 = 3/10 = 0.3
        assert result["per_query_results"][0]["precision_at_10"] == 0.3

    def test_ten_query_floor(self, harness_with_records):
        """Refuse to evaluate fewer than 10 queries."""
        queries = [AttributeSearchQuery("q1", {"upper_color": "red"}, ["t001"])] * 9  # Only 9

        with pytest.raises(ValueError, match="requires ≥10 queries"):
            harness_with_records.evaluate(queries)

    def test_deferred_status_no_threshold(self, harness_with_records):
        """Output shows DEFERRED, not threshold_met."""
        queries = [
            AttributeSearchQuery(
                "q1",
                {"upper_color": "red"},
                ["t001", "t002", "t003", "t004", "t005"],
            )
        ] * 10

        result = harness_with_records.evaluate(queries)

        assert result["status"] == "DEFERRED"
        assert "threshold_met" not in result
        assert result["ac08_threshold"] == 0.80  # Reference only

    def test_json_output(self, harness_with_records):
        """JSON output is valid."""
        queries = [AttributeSearchQuery("q1", {"upper_color": "red"}, ["t001", "t002"])] * 10

        json_str = harness_with_records.evaluate_to_json(queries)
        result = json.loads(json_str)

        assert "fixture_id" in result
        assert "status" in result
        assert result["status"] == "DEFERRED"
