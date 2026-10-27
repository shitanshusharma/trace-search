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

import warnings

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
        """Denim/navy detected as dark_blue."""
        frame = self.create_color_patch((90, 50, 60), size=(100, 50))
        attrs = extractor.extract(frame, (0, 0, 50, 100))
        assert attrs.upper_clothing_color == Color.DARK_BLUE
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
        crop[: int(0.25 * height), :] = (100, 100, 100)  # head
        crop[int(0.25 * height) : int(0.55 * height), :] = upper_bgr  # torso
        crop[int(0.55 * height) : int(0.90 * height), :] = lower_bgr  # legs
        crop[int(0.90 * height) :, :] = (100, 100, 100)  # feet
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
        """Negative coordinates clipped, then the visible blue pixels are read."""
        frame = np.full((100, 100, 3), (220, 0, 0), dtype=np.uint8)
        attrs = extractor.extract(frame, (-10, -10, 50, 50))
        # Box clipped to (0,0,50,50) on a solid-blue frame
        assert attrs.upper_clothing_color == Color.BLUE

    def test_out_of_frame_box(self, extractor):
        """Box extending past frame is clipped and the visible green is read."""
        frame = np.full((100, 100, 3), (0, 200, 0), dtype=np.uint8)
        attrs = extractor.extract(frame, (50, 50, 150, 150))
        # Box clipped to (50,50,100,100) on a solid-green frame
        assert attrs.upper_clothing_color == Color.GREEN

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

    def test_crowd_two_people_one_bag_inside_a(self, extractor):
        """Bag overlaps both people, mostly inside A; only A gets it."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        person_a_box = (10, 20, 90, 80)
        person_b_box = (70, 20, 150, 80)
        # Overlaps both: containment A = 0.875, B = 0.625 (both > 0.5)
        bag_box = (55, 30, 95, 60)

        assert AttributeExtractor._containment_ratio(person_a_box, bag_box) > 0.5
        assert AttributeExtractor._containment_ratio(person_b_box, bag_box) > 0.5
        assert AttributeExtractor._containment_ratio(
            person_a_box, bag_box
        ) > AttributeExtractor._containment_ratio(person_b_box, bag_box)

        assigned = AttributeExtractor.assign_bags([person_a_box, person_b_box], [bag_box])
        assert assigned == [True, False]

        attrs = extractor.extract_frame(frame, [person_a_box, person_b_box], [bag_box])
        assert attrs[0].bag_on_track is True
        assert attrs[1].bag_on_track is False

    def test_equal_straddle_tie_break(self, extractor):
        """Equal containment: earlier person in the list wins."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        person_a_box = (10, 20, 90, 80)
        person_b_box = (70, 20, 150, 80)
        # Review repro: containment is 0.833 for both people
        bag_box = (65, 30, 95, 60)

        containment_a = AttributeExtractor._containment_ratio(person_a_box, bag_box)
        containment_b = AttributeExtractor._containment_ratio(person_b_box, bag_box)
        assert containment_a == pytest.approx(containment_b)
        assert containment_a > 0.5

        assigned = AttributeExtractor.assign_bags([person_a_box, person_b_box], [bag_box])
        assert assigned == [True, False]

        assigned_reversed = AttributeExtractor.assign_bags([person_b_box, person_a_box], [bag_box])
        assert assigned_reversed == [True, False]

        attrs = extractor.extract_frame(frame, [person_a_box, person_b_box], [bag_box])
        assert attrs[0].bag_on_track is True
        assert attrs[1].bag_on_track is False

    def test_second_bag_stays_on_person_a(self, extractor):
        """A second bag mostly inside A does not spill onto neighbour B."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        person_a_box = (10, 20, 90, 80)
        person_b_box = (70, 20, 150, 80)
        bag1 = (20, 30, 40, 50)  # Fully inside A
        bag2 = (60, 30, 90, 60)  # Fully inside A, also overlaps B above 0.5

        assert AttributeExtractor._containment_ratio(person_a_box, bag1) == pytest.approx(1.0)
        assert AttributeExtractor._containment_ratio(person_b_box, bag1) == 0.0
        assert AttributeExtractor._containment_ratio(person_a_box, bag2) == pytest.approx(1.0)
        assert AttributeExtractor._containment_ratio(person_b_box, bag2) > 0.5
        assert AttributeExtractor._containment_ratio(
            person_a_box, bag2
        ) > AttributeExtractor._containment_ratio(person_b_box, bag2)

        assigned = AttributeExtractor.assign_bags([person_a_box, person_b_box], [bag1, bag2])
        assert assigned == [True, False]

        attrs = extractor.extract_frame(frame, [person_a_box, person_b_box], [bag1, bag2])
        assert attrs[0].bag_on_track is True
        assert attrs[1].bag_on_track is False

    def test_float_box_coordinates(self, extractor):
        """Float detector boxes are clipped and rounded."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        person_box = (20.7, 20.3, 80.9, 80.1)  # Float coords
        bag_boxes = [(30.5, 30.5, 60.8, 60.2)]  # Float coords

        # Floor origin / ceil opposite corner, then associate
        attrs = extractor.extract(frame, person_box, bag_boxes)
        # After floor/ceil: person=(20,20,81,81), bag=(30,30,61,61)
        assert attrs.bag_on_track is True


class TestDimLight:
    """Test low-light robustness."""

    def test_dim_red_shirt(self):
        """Dim red detected as red with reduced confidence."""
        extractor = AttributeExtractor(low_light_threshold=100)
        frame = np.full((200, 80, 3), (0, 0, 80), dtype=np.uint8)
        attrs = extractor.extract(frame, (0, 0, 80, 200))
        assert attrs.upper_clothing_color == Color.RED
        assert 0.3 < attrs.upper_confidence < 1.0


class TestHsvConversion:
    """HSV conversion must not warn on black or greyscale pixels."""

    def test_black_pixels_no_runtime_warning(self):
        """All-black frame must not emit divide-by-zero or invalid warnings."""
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            hsv = AttributeExtractor.bgr_to_hsv(frame)
        assert hsv.shape == frame.shape
        assert np.all(hsv[:, :, 2] == 0)


class TestSchemaValidation:
    """Verify no sensitive attributes in schema."""

    def test_no_sensitive_fields(self):
        """Schema must not contain gender, age, ethnicity, face, identity."""
        from dataclasses import fields

        field_names = {f.name for f in fields(PersonAttributes)}

        banned = {
            "gender",
            "sex",
            "is_male",
            "is_female",
            "age",
            "age_group",
            "ethnicity",
            "race",
            "skin_color",
            "face",
            "height",
            "height_cm",
            "pose",
            "keypoints",
            "name",
            "identity",
            "person_id",
            "id",
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


def _person(
    *,
    upper: Color = Color.RED,
    lower: Color = Color.BLUE,
    bag: bool = True,
    upper_confidence: float = 0.9,
    lower_confidence: float = 0.8,
) -> PersonAttributes:
    return PersonAttributes(
        upper_clothing_color=upper,
        lower_clothing_color=lower,
        bag_on_track=bag,
        upper_confidence=upper_confidence,
        lower_confidence=lower_confidence,
    )


class TestAttributeQueryValidation:
    """Present query fields are checked before a match can return False."""

    def test_bag_on_track_yes_raises_type_error(self):
        """A string bag flag is a type error, not a failed comparison."""
        with pytest.raises(TypeError, match="bag_on_track"):
            AttributeQuery.matches(_person(), {"bag_on_track": "yes"})

    @pytest.mark.parametrize(
        "value", [1, 0, None, np.bool_(True)], ids=["1", "0", "None", "np.bool_"]
    )
    def test_bag_on_track_int_raises_type_error(self, value: object):
        """0, 1, None, and np.bool_ are not bools, even when they compare equal."""
        with pytest.raises(TypeError, match="bag_on_track"):
            AttributeQuery.matches(_person(bag=bool(value)), {"bag_on_track": value})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("upper_color", "mauve"),
            ("lower_color", "mauve"),
            ("upper_color", "RED"),
            ("lower_color", " red"),
            ("upper_color", ""),
        ],
    )
    def test_bad_color_string_raises_value_error(self, key: str, value: str):
        """A string that is not a Color value is a ValueError naming the field."""
        with pytest.raises(ValueError, match=key):
            AttributeQuery.matches(_person(), {key: value})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("upper_color", 3),
            ("lower_color", 3),
            ("upper_color", None),
            ("lower_color", None),
        ],
    )
    def test_non_str_color_raises_type_error(self, key: str, value: object):
        """A non-str colour, including None, is a TypeError."""
        with pytest.raises(TypeError, match=key):
            AttributeQuery.matches(_person(), {key: value})

    def test_mauve_upper_raises_before_low_confidence_reject(self):
        """A bad upper colour raises even when confidence would already reject."""
        person = _person(upper_confidence=0.2)
        with pytest.raises(ValueError, match="upper_color") as caught:
            AttributeQuery.matches(person, {"upper_color": "mauve", "min_confidence": 1.0})
        assert "mauve" in str(caught.value)

    def test_mauve_lower_raises_before_low_confidence_reject(self):
        """A bad lower colour raises even when confidence would already reject."""
        person = _person(lower_confidence=0.2)
        with pytest.raises(ValueError, match="lower_color") as caught:
            AttributeQuery.matches(person, {"lower_color": "mauve", "min_confidence": 1.0})
        assert "mauve" in str(caught.value)

    def test_bad_bag_raises_before_color_mismatch_returns_false(self):
        """A bad bag flag raises even when an earlier colour would not match."""
        with pytest.raises(TypeError, match="bag_on_track"):
            AttributeQuery.matches(_person(), {"upper_color": "blue", "bag_on_track": "yes"})

    def test_bad_bag_raises_before_lower_mismatch_returns_false(self):
        """A bad bag flag raises even when the lower colour would not match."""
        with pytest.raises(TypeError, match="bag_on_track"):
            AttributeQuery.matches(_person(), {"lower_color": "red", "bag_on_track": "yes"})

    def test_bad_lower_color_raises_before_upper_mismatch_returns_false(self):
        """A bad lower colour raises even when the upper colour would not match."""
        with pytest.raises(ValueError, match="lower_color"):
            AttributeQuery.matches(_person(), {"upper_color": "blue", "lower_color": "mauve"})

    def test_bad_lower_raises_before_upper_confidence_reject(self):
        """A bad lower colour raises even when upper confidence would reject."""
        person = _person(upper_confidence=0.2)
        with pytest.raises(ValueError, match="lower_color"):
            AttributeQuery.matches(
                person,
                {"upper_color": "red", "lower_color": "mauve", "min_confidence": 1.0},
            )

    def test_bad_bag_raises_before_confidence_reject(self):
        """A bad bag flag raises even when colour confidence would reject."""
        person = _person(upper_confidence=0.2, lower_confidence=0.2)
        with pytest.raises(TypeError, match="bag_on_track"):
            AttributeQuery.matches(
                person,
                {
                    "upper_color": "red",
                    "lower_color": "blue",
                    "bag_on_track": "yes",
                    "min_confidence": 1.0,
                },
            )

    def test_min_confidence_str_raises_type_error(self):
        """A string threshold is a TypeError, including when no colour is queried."""
        with pytest.raises(TypeError, match="min_confidence"):
            AttributeQuery.matches(_person(), {"min_confidence": "0.5"})

    @pytest.mark.parametrize(
        "value",
        [np.float32(0.5), np.int64(0), None],
        ids=["np.float32", "np.int64", "None"],
    )
    def test_min_confidence_numpy_or_none_raises_type_error(self, value: object):
        """numpy scalars that are not int or float, and None, are a TypeError."""
        with pytest.raises(TypeError, match="min_confidence"):
            AttributeQuery.matches(_person(), {"min_confidence": value})

    @pytest.mark.parametrize("value", [True, False])
    def test_min_confidence_bool_raises_type_error(self, value: bool):
        """bool is an int subclass and is still refused."""
        with pytest.raises(TypeError, match="min_confidence"):
            AttributeQuery.matches(_person(), {"min_confidence": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_min_confidence_non_finite_raises_value_error(self, value: float):
        """NaN and infinities are bad values, not a silent match."""
        with pytest.raises(ValueError, match="finite"):
            AttributeQuery.matches(_person(), {"upper_color": "red", "min_confidence": value})

    @pytest.mark.parametrize("value", [-0.1, 1.5, 2])
    def test_min_confidence_out_of_range_raises_value_error(self, value: float):
        """A number outside the closed range [0, 1] is a ValueError."""
        with pytest.raises(ValueError, match=r"min_confidence must be in \[0, 1\]"):
            AttributeQuery.matches(_person(), {"min_confidence": value})

    def test_out_of_range_min_confidence_raises_before_confidence_reject(self):
        """An out-of-range threshold raises instead of rejecting the colour."""
        person = _person(upper_confidence=0.2)
        with pytest.raises(ValueError, match="min_confidence"):
            AttributeQuery.matches(person, {"upper_color": "red", "min_confidence": 1.5})

    def test_min_confidence_type_is_checked_before_color_value(self):
        """The first invalid field in documented order raises."""
        with pytest.raises(TypeError, match="min_confidence"):
            AttributeQuery.matches(_person(), {"upper_color": "mauve", "min_confidence": "high"})

    def test_upper_color_is_checked_before_lower_color(self):
        """A bad upper colour wins over a bad lower colour of a different type."""
        with pytest.raises(ValueError, match="upper_color"):
            AttributeQuery.matches(_person(), {"upper_color": "mauve", "lower_color": 3})

    def test_lower_color_is_checked_before_bag(self):
        """A bad lower colour wins over a bad bag flag."""
        with pytest.raises(ValueError, match="lower_color"):
            AttributeQuery.matches(_person(), {"lower_color": "mauve", "bag_on_track": "yes"})

    def test_valid_query_regression(self):
        """Known-good queries keep the same accept and reject results."""
        person = _person(upper_confidence=0.5, lower_confidence=0.4)
        assert AttributeQuery.matches(person, {}) is True
        assert AttributeQuery.matches(person, {"not_a_field": "ignored"}) is True
        assert AttributeQuery.matches(person, {"upper_color": "red"}) is True
        assert AttributeQuery.matches(person, {"upper_color": Color.RED}) is True
        assert AttributeQuery.matches(person, {"upper_color": "blue"}) is False
        assert AttributeQuery.matches(person, {"lower_color": "blue"}) is True
        assert AttributeQuery.matches(person, {"lower_color": "red"}) is False
        assert AttributeQuery.matches(person, {"bag_on_track": True}) is True
        assert AttributeQuery.matches(person, {"bag_on_track": False}) is False
        assert AttributeQuery.matches(person, {"upper_color": "red", "min_confidence": 0.5}) is True
        assert (
            AttributeQuery.matches(person, {"lower_color": "blue", "min_confidence": 0.4}) is True
        )
        assert (
            AttributeQuery.matches(person, {"upper_color": "red", "min_confidence": 0.51}) is False
        )
        assert (
            AttributeQuery.matches(person, {"lower_color": "blue", "min_confidence": 0.41}) is False
        )
        assert AttributeQuery.matches(person, {"min_confidence": 0}) is True
        assert AttributeQuery.matches(person, {"min_confidence": 1}) is True
        assert AttributeQuery.matches(person, {"min_confidence": 0.0}) is True
        assert AttributeQuery.matches(person, {"min_confidence": 1.0}) is True
        assert AttributeQuery.matches(person, {"min_confidence": np.float64(0.4)}) is True
        assert AttributeQuery.matches(person, {"bag_on_track": True, "min_confidence": 1.0}) is True
        assert (
            AttributeQuery.matches(
                person,
                {
                    "upper_color": "red",
                    "lower_color": "blue",
                    "bag_on_track": True,
                    "min_confidence": 0.4,
                },
            )
            is True
        )
        low = _person(upper_confidence=0.2, lower_confidence=0.2, bag=False)
        assert AttributeQuery.matches(low, {"upper_color": "blue", "min_confidence": 1.0}) is False
        exact = _person(upper_confidence=1.0, lower_confidence=0.0)
        assert AttributeQuery.matches(exact, {"upper_color": "red", "min_confidence": 1.0}) is True
        assert AttributeQuery.matches(exact, {"lower_color": "blue", "min_confidence": 0}) is True
        assert (
            AttributeQuery.matches(exact, {"lower_color": "blue", "min_confidence": 0.01}) is False
        )
        both = _person(upper_confidence=0.9, lower_confidence=0.2)
        assert (
            AttributeQuery.matches(
                both,
                {"upper_color": "red", "lower_color": "blue", "min_confidence": 0.5},
            )
            is False
        )
