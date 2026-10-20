"""
Tests for attribute extraction (T4).

Covers:
- All palette colors on synthetic patches (seeded)
- Synthetic person crops
- 'invalid_box' and 'unknown' paths
- Correct bag association with crowd test
- Dim-light case
- Schema validation (no sensitive attributes)
- Negative coordinates and out-of-frame boxes
"""

import numpy as np
import pytest

from trace_search.attributes import (
    AttributeExtractor,
    AttributeQuery,
    Color,
    PersonAttributes,
)

# Local RNG for reproducibility
_rng = np.random.default_rng(42)


class TestColorPalette:
    """Test color classification on synthetic patches (seeded)."""

    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()

    def create_color_patch(self, bgr: tuple, size=(100, 50)) -> np.ndarray:
        """Create a solid color patch."""
        h, w = size
        patch = np.full((h, w, 3), bgr, dtype=np.uint8)
        return patch

    def test_red_patch(self, extractor):
        """Red clothing detected as red."""
        frame = self.create_color_patch((0, 0, 220), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.RED
        assert attrs.upper_confidence > 0.5

    def test_blue_patch(self, extractor):
        """Blue clothing detected as blue."""
        frame = self.create_color_patch((220, 0, 0), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.BLUE
        assert attrs.upper_confidence > 0.5

    def test_dark_blue_denim(self, extractor):
        """Denim/navy detected as dark_blue or blue."""
        frame = self.create_color_patch((90, 50, 60), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        # Dark blue has lower value than bright blue
        assert attrs.upper_clothing_color in [Color.DARK_BLUE, Color.BLUE]
        assert attrs.upper_confidence > 0.3

    def test_green_patch(self, extractor):
        """Green clothing detected as green."""
        frame = self.create_color_patch((0, 200, 0), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.GREEN
        assert attrs.upper_confidence > 0.5

    def test_yellow_patch(self, extractor):
        """Yellow clothing detected as yellow."""
        frame = self.create_color_patch((0, 220, 220), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.YELLOW
        assert attrs.upper_confidence > 0.5

    def test_black_patch(self, extractor):
        """Black clothing detected as black."""
        frame = self.create_color_patch((20, 20, 20), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.BLACK
        assert attrs.upper_confidence > 0.3

    def test_orange_patch(self, extractor):
        """Orange clothing detected as orange."""
        frame = self.create_color_patch((0, 120, 240), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.ORANGE
        assert attrs.upper_confidence > 0.5

    def test_cyan_patch(self, extractor):
        """Cyan clothing detected as cyan."""
        frame = self.create_color_patch((220, 220, 0), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.CYAN
        assert attrs.upper_confidence > 0.5

    def test_purple_patch(self, extractor):
        """Purple clothing detected as purple."""
        frame = self.create_color_patch((200, 0, 180), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.PURPLE
        assert attrs.upper_confidence > 0.5

    def test_pink_patch(self, extractor):
        """Pink clothing detected as pink."""
        frame = self.create_color_patch((180, 100, 255), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.PINK
        assert attrs.upper_confidence > 0.5

    def test_brown_patch(self, extractor):
        """Brown clothing detected as brown."""
        # Brown: low saturation orange (BGR: moderate blue, higher green/red)
        frame = self.create_color_patch((60, 75, 90), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.BROWN
        assert attrs.upper_confidence > 0.3

    def test_gray_patch(self, extractor):
        """Gray clothing detected as gray."""
        frame = self.create_color_patch((120, 120, 120), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.GRAY
        assert attrs.upper_confidence > 0.5

    def test_white_patch(self, extractor):
        """White clothing detected as white."""
        frame = self.create_color_patch((240, 240, 240), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.WHITE
        assert attrs.upper_confidence > 0.5


class TestSyntheticPerson:
    """Test on synthetic person crops with torso and leg regions."""

    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()

    def create_person_crop(
        self, height: int, width: int, upper_bgr: tuple, lower_bgr: tuple
    ) -> np.ndarray:
        """
        Create synthetic person with colored regions.

        Layout:
        - Top 25%: head (gray)
        - 25-55%: torso (upper_bgr)
        - 55-90%: legs (lower_bgr)
        - Bottom 10%: feet (gray)
        """
        crop = np.zeros((height, width, 3), dtype=np.uint8)
        crop[:int(0.25 * height), :] = (100, 100, 100)  # head
        crop[int(0.25 * height):int(0.55 * height), :] = upper_bgr  # torso
        crop[int(0.55 * height):int(0.90 * height), :] = lower_bgr  # legs
        crop[int(0.90 * height):, :] = (100, 100, 100)  # feet
        return crop

    def test_red_shirt_blue_pants(self, extractor):
        """Red shirt + blue pants correctly detected."""
        frame = self.create_person_crop(200, 80, (0, 0, 220), (220, 0, 0))
        attrs = extractor.extract(frame, (0, 0, 80, 200))
        assert attrs.upper_clothing_color == Color.RED
        assert attrs.lower_clothing_color == Color.BLUE
        assert attrs.upper_confidence > 0.5
        assert attrs.lower_confidence > 0.5

    def test_white_shirt_black_pants(self, extractor):
        """White shirt + black pants correctly detected."""
        frame = self.create_person_crop(200, 80, (240, 240, 240), (20, 20, 20))
        attrs = extractor.extract(frame, (0, 0, 80, 200))
        assert attrs.upper_clothing_color == Color.WHITE
        assert attrs.lower_clothing_color == Color.BLACK


class TestInvalidBox:
    """Test INVALID_BOX for degenerate and out-of-frame boxes."""

    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()

    def test_negative_coordinates(self, extractor):
        """Negative coordinates clipped, then processed."""
        frame = np.full((100, 100, 3), (220, 0, 0), dtype=np.uint8)
        attrs = extractor.extract(frame, (-10, -10, 50, 50))
        # Box clipped to (0,0,50,50), should still work
        assert attrs.upper_clothing_color != Color.INVALID_BOX

    def test_out_of_frame_box(self, extractor):
        """Box extending past frame is clipped."""
        frame = np.full((100, 100, 3), (0, 200, 0), dtype=np.uint8)
        attrs = extractor.extract(frame, (50, 50, 150, 150))
        # Box clipped to (50,50,100,100), should still work
        assert attrs.upper_clothing_color != Color.INVALID_BOX

    def test_zero_area_box(self, extractor):
        """Zero-area box returns INVALID_BOX."""
        frame = _rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        attrs = extractor.extract(frame, (50, 50, 50, 50))
        assert attrs.upper_clothing_color == Color.INVALID_BOX
        assert attrs.lower_clothing_color == Color.INVALID_BOX
        assert attrs.upper_confidence == 0.0

    def test_inverted_box(self, extractor):
        """Inverted box (x2 < x1) returns INVALID_BOX."""
        frame = _rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        attrs = extractor.extract(frame, (80, 80, 20, 20))
        assert attrs.upper_clothing_color == Color.INVALID_BOX


class TestBagDetection:
    """Test bag detection with containment-based assignment."""

    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()

    def test_bag_fully_inside_person(self, extractor):
        """Backpack fully inside person box is assigned (containment=1.0)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 80, 80)
        bag_boxes = [(30, 30, 60, 60)]  # Fully inside person
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.bag_on_track is True

    def test_bag_overlap_high_containment(self, extractor):
        """High containment bag assigned to track."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 80, 80)
        bag_boxes = [(25, 25, 70, 70)]  # Most of bag inside person
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.bag_on_track is True

    def test_no_bag(self, extractor):
        """No bag boxes."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        attrs = extractor.extract(frame, (20, 20, 80, 80), bag_boxes=None)
        assert attrs.bag_on_track is False

    def test_bag_low_containment(self, extractor):
        """Low containment bag not assigned."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 40, 40)
        bag_boxes = [(35, 35, 70, 70)]  # Only small portion inside person
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.bag_on_track is False

    def test_crowd_two_people_one_bag_inside_A(self, extractor):
        """Two overlapping people, one bag clearly inside person A."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        person_a_box = (10, 20, 90, 80)  # Left person
        person_b_box = (70, 20, 150, 80)  # Right person (overlaps)
        bag_box = (30, 35, 50, 55)  # Clearly inside person A

        # Bag is fully inside person A
        attrs_a = extractor.extract(frame, person_a_box, [bag_box])
        assert attrs_a.bag_on_track is True

        # Bag has low containment in person B (only partial overlap)
        attrs_b = extractor.extract(frame, person_b_box, [bag_box])
        assert attrs_b.bag_on_track is False

    def test_float_box_coordinates(self, extractor):
        """Float detector boxes are clipped and rounded."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20.7, 20.3, 80.9, 80.1)  # Float coords
        bag_boxes = [(30.5, 30.5, 60.8, 60.2)]  # Float coords

        # Should clip, round to int, then process
        attrs = extractor.extract(frame, person_box, bag_boxes)
        # After rounding: person=(21,20,81,80), bag=(31,31,61,60)
        # Bag mostly inside, should be assigned
        assert attrs.bag_on_track is True


class TestDimLight:
    """Test low-light robustness."""

    def test_dim_red_shirt(self):
        """Dim red detected with reduced confidence."""
        extractor = AttributeExtractor(low_light_threshold=100)
        frame = np.full((200, 80, 3), (0, 0, 80), dtype=np.uint8)
        attrs = extractor.extract(frame, (0, 0, 80, 200))
        assert attrs.upper_clothing_color in [Color.RED, Color.UNKNOWN, Color.DARK_BLUE]


class TestSchemaValidation:
    """Verify no sensitive attributes in schema."""

    def test_no_sensitive_fields(self):
        """Schema must not contain gender, age, ethnicity, face, identity."""
        from dataclasses import fields

        field_names = {f.name for f in fields(PersonAttributes)}

        banned = {
            "gender", "sex", "is_male", "is_female",
            "age", "age_group",
            "ethnicity", "race", "skin_color",
            "face", "face_embedding", "face_crop",
            "height", "height_cm",
            "pose", "keypoints",
            "name", "identity", "person_id", "id",
        }

        overlap = field_names & banned
        assert len(overlap) == 0, f"Found banned fields: {overlap}"

    def test_only_allowed_fields(self):
        """Schema contains only allowed appearance attributes."""
        from dataclasses import fields

        field_names = {f.name for f in fields(PersonAttributes)}

        allowed = {
            "upper_clothing_color",
            "lower_clothing_color",
            "bag_on_track",
            "upper_confidence",
            "lower_confidence",
        }

        assert field_names == allowed, f"Unexpected fields: {field_names - allowed}"


class TestAttributeQuery:
    """Test query helper."""

    def test_bag_query(self):
        """Query by bag_on_track."""
        attrs = PersonAttributes(
            Color.RED, Color.BLUE, bag_on_track=True, upper_confidence=0.9, lower_confidence=0.8
        )

        assert AttributeQuery.matches(attrs, {"bag_on_track": True}) is True
        assert AttributeQuery.matches(attrs, {"bag_on_track": False}) is False

    def test_combined_query(self):
        """Combined query with AND logic."""
        attrs = PersonAttributes(
            Color.RED, Color.BLUE, bag_on_track=True, upper_confidence=0.9, lower_confidence=0.8
        )

        assert AttributeQuery.matches(attrs, {"upper_color": "red", "bag_on_track": True}) is True
        assert AttributeQuery.matches(attrs, {"upper_color": "red", "bag_on_track": False}) is False
