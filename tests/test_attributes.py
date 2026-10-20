"""
Tests for attribute extraction (T4).

Covers:
- Synthetic color patches (all palette colors)
- Synthetic person crops
- 'unknown' path for low confidence
- Correct bag association
- Dim-light case
- Schema validation (no sensitive attributes)
"""

import pytest
import numpy as np
import cv2
from trace_search.attributes import (
    AttributeExtractor,
    PersonAttributes,
    Color,
    AttributeQuery,
)


class TestColorPalette:
    """Test color classification on synthetic patches."""
    
    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()
    
    def create_color_patch(self, bgr: tuple, size=(100, 50)) -> np.ndarray:
        """Create a solid color patch."""
        h, w = size
        patch = np.full((h, w, 3), bgr, dtype=np.uint8)
        return patch
    
    def test_red_patch(self, extractor):
        """Red clothing should be detected as red."""
        frame = self.create_color_patch((0, 0, 220), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.RED
        assert attrs.upper_confidence > 0.5
    
    def test_blue_patch(self, extractor):
        """Blue clothing should be detected as blue."""
        frame = self.create_color_patch((220, 0, 0), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.BLUE
        assert attrs.upper_confidence > 0.5
    
    def test_green_patch(self, extractor):
        """Green clothing should be detected as green."""
        frame = self.create_color_patch((0, 200, 0), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.GREEN
        assert attrs.upper_confidence > 0.5
    
    def test_yellow_patch(self, extractor):
        """Yellow clothing should be detected as yellow."""
        frame = self.create_color_patch((0, 220, 220), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.YELLOW
        assert attrs.upper_confidence > 0.5
    
    def test_orange_patch(self, extractor):
        """Orange clothing should be detected as orange."""
        frame = self.create_color_patch((0, 120, 240), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.ORANGE
        assert attrs.upper_confidence > 0.5
    
    def test_purple_patch(self, extractor):
        """Purple clothing should be detected as purple."""
        frame = self.create_color_patch((200, 0, 180), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.PURPLE
        assert attrs.upper_confidence > 0.5
    
    def test_cyan_patch(self, extractor):
        """Cyan clothing should be detected as cyan."""
        frame = self.create_color_patch((220, 220, 0), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.CYAN
        assert attrs.upper_confidence > 0.5
    
    def test_pink_patch(self, extractor):
        """Pink clothing should be detected as pink."""
        frame = self.create_color_patch((180, 100, 255), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.PINK
        assert attrs.upper_confidence > 0.5
    
    def test_black_patch(self, extractor):
        """Black clothing should be detected as black."""
        frame = self.create_color_patch((20, 20, 20), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.BLACK
        assert attrs.upper_confidence > 0.5
    
    def test_white_patch(self, extractor):
        """White clothing should be detected as white."""
        frame = self.create_color_patch((240, 240, 240), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.WHITE
        assert attrs.upper_confidence > 0.5
    
    def test_gray_patch(self, extractor):
        """Gray clothing should be detected as gray."""
        frame = self.create_color_patch((120, 120, 120), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.GRAY
        assert attrs.upper_confidence > 0.5
    
    def test_brown_patch(self, extractor):
        """Brown clothing should be detected as brown."""
        # Brown: desaturated orange (low sat, medium hue, medium value)
        frame = self.create_color_patch((60, 75, 90), size=(100, 50))
        person_box = (0, 0, 50, 100)
        
        attrs = extractor.extract(frame, person_box)
        # Brown is tricky - may be orange or brown depending on saturation
        assert attrs.upper_clothing_color in [Color.BROWN, Color.ORANGE]
        assert attrs.upper_confidence > 0.3


class TestSyntheticPerson:
    """Test on synthetic person crops with torso and leg regions."""
    
    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()
    
    def create_person_crop(
        self,
        height: int,
        width: int,
        upper_color_bgr: tuple,
        lower_color_bgr: tuple,
    ) -> np.ndarray:
        """
        Create a synthetic person crop with colored regions.
        
        Layout:
        - Top 20%: head (gray)
        - 20-60%: torso (upper_color)
        - 60-100%: legs (lower_color)
        """
        crop = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Head region (top 20%)
        head_end = int(0.2 * height)
        crop[:head_end, :] = (100, 100, 100)  # gray head
        
        # Torso region (20-60%)
        torso_start = head_end
        torso_end = int(0.6 * height)
        crop[torso_start:torso_end, :] = upper_color_bgr
        
        # Legs region (60-100%)
        legs_start = torso_end
        crop[legs_start:, :] = lower_color_bgr
        
        return crop
    
    def test_red_shirt_blue_pants(self, extractor):
        """Person with red shirt and blue pants."""
        frame = self.create_person_crop(
            height=200,
            width=80,
            upper_color_bgr=(0, 0, 220),  # red
            lower_color_bgr=(220, 0, 0),  # blue
        )
        person_box = (0, 0, 80, 200)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.RED
        assert attrs.lower_clothing_color == Color.BLUE
        assert attrs.upper_confidence > 0.5
        assert attrs.lower_confidence > 0.5
    
    def test_white_shirt_black_pants(self, extractor):
        """Person with white shirt and black pants."""
        frame = self.create_person_crop(
            height=200,
            width=80,
            upper_color_bgr=(240, 240, 240),  # white
            lower_color_bgr=(20, 20, 20),  # black
        )
        person_box = (0, 0, 80, 200)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.WHITE
        assert attrs.lower_clothing_color == Color.BLACK
    
    def test_green_shirt_brown_pants(self, extractor):
        """Person with green shirt and brown/orange pants."""
        frame = self.create_person_crop(
            height=200,
            width=80,
            upper_color_bgr=(0, 200, 0),  # green
            lower_color_bgr=(60, 75, 90),  # brownish
        )
        person_box = (0, 0, 80, 200)
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.GREEN
        # Brown detection is ambiguous; orange is also acceptable
        assert attrs.lower_clothing_color in [Color.BROWN, Color.ORANGE, Color.GRAY]
        assert attrs.lower_confidence > 0.2


class TestUnknownPath:
    """Test that unknown is returned for low confidence cases."""
    
    @pytest.fixture
    def extractor(self):
        return AttributeExtractor(min_confidence=0.5)
    
    def test_very_small_box(self, extractor):
        """Tiny person box should return unknown."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        person_box = (0, 0, 5, 10)  # Too small
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.UNKNOWN
        assert attrs.lower_clothing_color == Color.UNKNOWN
        assert attrs.upper_confidence < 0.5
    
    def test_empty_box(self, extractor):
        """Empty/invalid box should return unknown."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        person_box = (50, 50, 50, 50)  # Zero area
        
        attrs = extractor.extract(frame, person_box)
        assert attrs.upper_clothing_color == Color.UNKNOWN
        assert attrs.lower_clothing_color == Color.UNKNOWN
        assert attrs.upper_confidence == 0.0
    
    def test_highly_mixed_colors(self, extractor):
        """Noisy multi-color region may return unknown."""
        # Create checkerboard pattern
        frame = np.zeros((200, 80, 3), dtype=np.uint8)
        for i in range(0, 200, 10):
            for j in range(0, 80, 10):
                color = np.random.randint(0, 255, 3)
                frame[i:i+10, j:j+10] = color
        
        person_box = (0, 0, 80, 200)
        attrs = extractor.extract(frame, person_box)
        
        # At least one should be low confidence due to noise
        assert (
            attrs.upper_confidence < 0.5 or
            attrs.lower_confidence < 0.5 or
            attrs.upper_clothing_color == Color.UNKNOWN or
            attrs.lower_clothing_color == Color.UNKNOWN
        )


class TestDimLight:
    """Test behavior in low-light conditions."""
    
    def test_dim_red_shirt(self):
        """Dim red should still be detected but with lower confidence."""
        extractor = AttributeExtractor(low_light_threshold=100)
        
        # Create dim red (low value)
        frame = np.full((200, 80, 3), (0, 0, 80), dtype=np.uint8)  # Dim red
        person_box = (0, 0, 80, 200)
        
        attrs = extractor.extract(frame, person_box)
        
        # Should still detect red but with reduced confidence
        assert attrs.upper_clothing_color == Color.RED or attrs.upper_clothing_color == Color.UNKNOWN
        
        # Confidence should be reduced due to dim lighting
        if attrs.upper_clothing_color == Color.RED:
            assert attrs.upper_confidence < 0.9  # Penalized
    
    def test_very_dark_scene(self):
        """Very dark scene should return unknown or black."""
        extractor = AttributeExtractor()
        
        frame = np.full((200, 80, 3), (10, 10, 10), dtype=np.uint8)  # Nearly black
        person_box = (0, 0, 80, 200)
        
        attrs = extractor.extract(frame, person_box)
        
        # Should be black or unknown
        assert attrs.upper_clothing_color in [Color.BLACK, Color.UNKNOWN]


class TestBagDetection:
    """Test bag attribute from detector boxes."""
    
    @pytest.fixture
    def extractor(self):
        return AttributeExtractor()
    
    def test_bag_overlap(self, extractor):
        """Bag box overlapping person should set has_bag=True."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 80, 80)
        bag_boxes = [(30, 30, 60, 60)]  # Overlaps person
        
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.has_bag is True
    
    def test_no_bag(self, extractor):
        """No bag boxes should set has_bag=False."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 80, 80)
        
        attrs = extractor.extract(frame, person_box, bag_boxes=None)
        assert attrs.has_bag is False
    
    def test_bag_no_overlap(self, extractor):
        """Bag box not overlapping person should set has_bag=False."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 40, 40)
        bag_boxes = [(60, 60, 80, 80)]  # No overlap
        
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.has_bag is False
    
    def test_multiple_bags_one_overlaps(self, extractor):
        """Multiple bags, one overlapping, should set has_bag=True."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20, 20, 80, 80)
        bag_boxes = [
            (0, 0, 10, 10),  # No overlap
            (30, 30, 60, 60),  # Overlaps
            (90, 90, 95, 95),  # No overlap
        ]
        
        attrs = extractor.extract(frame, person_box, bag_boxes)
        assert attrs.has_bag is True


class TestSchemaValidation:
    """Test that PersonAttributes contains no sensitive fields."""
    
    def test_no_sensitive_fields(self):
        """Schema must not contain gender, age, ethnicity, face, height."""
        from dataclasses import fields
        
        field_names = {f.name for f in fields(PersonAttributes)}
        
        # Banned sensitive attributes
        banned = {
            "gender", "sex", "is_male", "is_female",
            "age", "age_group",
            "ethnicity", "race", "skin_color",
            "face", "face_embedding", "face_crop",
            "height", "height_cm",
            "pose", "keypoints",
            "name", "identity", "person_id",
        }
        
        overlap = field_names & banned
        assert len(overlap) == 0, f"Found banned fields: {overlap}"
    
    def test_only_allowed_fields(self):
        """Schema must contain only the allowed appearance attributes."""
        from dataclasses import fields
        
        field_names = {f.name for f in fields(PersonAttributes)}
        
        allowed = {
            "upper_clothing_color",
            "lower_clothing_color",
            "has_bag",
            "upper_confidence",
            "lower_confidence",
        }
        
        assert field_names == allowed, f"Unexpected fields: {field_names - allowed}"


class TestAttributeQuery:
    """Test query helper."""
    
    def test_upper_color_match(self):
        """Query by upper color."""
        attrs = PersonAttributes(
            upper_clothing_color=Color.RED,
            lower_clothing_color=Color.BLUE,
            has_bag=False,
            upper_confidence=0.9,
            lower_confidence=0.8,
        )
        
        query = {"upper_color": "red"}
        assert AttributeQuery.matches(attrs, query) is True
        
        query = {"upper_color": "blue"}
        assert AttributeQuery.matches(attrs, query) is False
    
    def test_lower_color_match(self):
        """Query by lower color."""
        attrs = PersonAttributes(
            upper_clothing_color=Color.RED,
            lower_clothing_color=Color.BLUE,
            has_bag=False,
            upper_confidence=0.9,
            lower_confidence=0.8,
        )
        
        query = {"lower_color": "blue"}
        assert AttributeQuery.matches(attrs, query) is True
        
        query = {"lower_color": "red"}
        assert AttributeQuery.matches(attrs, query) is False
    
    def test_bag_match(self):
        """Query by bag presence."""
        attrs = PersonAttributes(
            upper_clothing_color=Color.RED,
            lower_clothing_color=Color.BLUE,
            has_bag=True,
            upper_confidence=0.9,
            lower_confidence=0.8,
        )
        
        query = {"has_bag": True}
        assert AttributeQuery.matches(attrs, query) is True
        
        query = {"has_bag": False}
        assert AttributeQuery.matches(attrs, query) is False
    
    def test_combined_query(self):
        """Query with multiple conditions (AND logic)."""
        attrs = PersonAttributes(
            upper_clothing_color=Color.RED,
            lower_clothing_color=Color.BLUE,
            has_bag=True,
            upper_confidence=0.9,
            lower_confidence=0.8,
        )
        
        query = {"upper_color": "red", "has_bag": True}
        assert AttributeQuery.matches(attrs, query) is True
        
        query = {"upper_color": "red", "has_bag": False}
        assert AttributeQuery.matches(attrs, query) is False
        
        query = {"upper_color": "red", "lower_color": "blue", "has_bag": True}
        assert AttributeQuery.matches(attrs, query) is True
    
    def test_confidence_threshold(self):
        """Query with minimum confidence requirement."""
        attrs = PersonAttributes(
            upper_clothing_color=Color.RED,
            lower_clothing_color=Color.BLUE,
            has_bag=False,
            upper_confidence=0.4,
            lower_confidence=0.9,
        )
        
        # Upper confidence too low
        query = {"upper_color": "red", "min_confidence": 0.5}
        assert AttributeQuery.matches(attrs, query) is False
        
        # Lower confidence OK
        query = {"lower_color": "blue", "min_confidence": 0.5}
        assert AttributeQuery.matches(attrs, query) is True
