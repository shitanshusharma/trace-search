"""
Deterministic person attribute extraction (T4).

Extracts upper-clothing color, lower-clothing color, and bag presence
from person bounding boxes. No trained weights, no sensitive attributes.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Color(str, Enum):
    """Named color palette for clothing."""

    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PersonAttributes:
    """
    Person appearance attributes.

    Contains ONLY non-sensitive attributes:
    - upper_clothing_color
    - lower_clothing_color
    - has_bag
    - upper_confidence
    - lower_confidence

    Explicitly excludes: gender, age, ethnicity, face features, height, pose.
    """

    upper_clothing_color: Color
    lower_clothing_color: Color
    has_bag: bool
    upper_confidence: float
    lower_confidence: float

    def __post_init__(self):
        assert 0.0 <= self.upper_confidence <= 1.0
        assert 0.0 <= self.lower_confidence <= 1.0


class AttributeExtractor:
    """
    Deterministic attribute extractor using HSV color binning.

    Strategy:
    - Divide person box into torso (upper 40%) and legs (lower 40%)
    - Skip head region (top 20%) and feet region (bottom 20%)
    - Convert to HSV and bin dominant hue/saturation/value
    - Return 'unknown' when confidence is low
    - Handle low-light via value channel thresholding
    """

    # HSV color boundaries (H: 0-180, S: 0-255, V: 0-255 in OpenCV)
    COLOR_RANGES = {
        Color.RED: [
            (0, 100, 50),
            (10, 255, 255),
            (170, 100, 50),
            (180, 255, 255),
        ],  # Red wraps around
        Color.ORANGE: [(11, 100, 50), (25, 255, 255)],
        Color.YELLOW: [(26, 100, 50), (35, 255, 255)],
        Color.GREEN: [(36, 100, 50), (85, 255, 255)],
        Color.CYAN: [(86, 100, 50), (100, 255, 255)],
        Color.BLUE: [(101, 100, 50), (130, 255, 255)],
        Color.PURPLE: [(131, 100, 50), (160, 255, 255)],
        Color.PINK: [(161, 100, 50), (169, 255, 255)],
    }

    # Low saturation -> grayscale
    GRAY_THRESHOLD_SAT = 40

    # Value thresholds for black/white
    BLACK_THRESHOLD_VAL = 50
    WHITE_THRESHOLD_VAL = 200

    # Brown: low saturation orange/yellow with medium value
    BROWN_HUE_RANGE = (11, 35)
    BROWN_SAT_RANGE = (40, 100)
    BROWN_VAL_RANGE = (50, 150)

    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    LOW_LIGHT_VALUE = 80
    MIN_PIXELS_RATIO = 0.1  # Region must have at least 10% valid pixels

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        low_light_threshold: int = LOW_LIGHT_VALUE,
    ):
        self.min_confidence = min_confidence
        self.low_light_threshold = low_light_threshold

    def extract(
        self,
        frame: np.ndarray,
        person_box: tuple[int, int, int, int],
        bag_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> PersonAttributes:
        """
        Extract attributes from a person bounding box.

        Args:
            frame: BGR image (H, W, 3)
            person_box: (x1, y1, x2, y2) in pixel coordinates
            bag_boxes: Optional list of (x1, y1, x2, y2) backpack/handbag boxes

        Returns:
            PersonAttributes with color and bag information
        """
        x1, y1, x2, y2 = person_box
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return PersonAttributes(
                upper_clothing_color=Color.UNKNOWN,
                lower_clothing_color=Color.UNKNOWN,
                has_bag=False,
                upper_confidence=0.0,
                lower_confidence=0.0,
            )

        # Extract color from torso and leg regions
        upper_color, upper_conf = self._extract_region_color(
            person_crop, region="upper"
        )
        lower_color, lower_conf = self._extract_region_color(
            person_crop, region="lower"
        )

        # Determine bag presence from overlapping detector boxes
        has_bag = self._has_bag(person_box, bag_boxes)

        return PersonAttributes(
            upper_clothing_color=upper_color,
            lower_clothing_color=lower_color,
            has_bag=has_bag,
            upper_confidence=upper_conf,
            lower_confidence=lower_conf,
        )

    def _extract_region_color(
        self,
        person_crop: np.ndarray,
        region: str,
    ) -> tuple[Color, float]:
        """
        Extract dominant color from torso or leg region.

        Args:
            person_crop: BGR person crop
            region: "upper" (torso) or "lower" (legs)

        Returns:
            (color, confidence) tuple
        """
        h, w = person_crop.shape[:2]

        if h < 20 or w < 10:
            return Color.UNKNOWN, 0.0

        # Define regions (skip head and feet)
        if region == "upper":
            # Torso: 20% to 60% of height
            y_start = int(0.2 * h)
            y_end = int(0.6 * h)
        else:  # lower
            # Legs: 60% to 100% of height (but often occluded at bottom)
            y_start = int(0.6 * h)
            y_end = h

        region_crop = person_crop[y_start:y_end, :]

        if region_crop.size == 0:
            return Color.UNKNOWN, 0.0

        # Convert to HSV
        hsv = cv2.cvtColor(region_crop, cv2.COLOR_BGR2HSV)

        # Classify color
        color, base_confidence = self._classify_color_hsv(hsv)

        # Check for low-light conditions
        # For grayscale colors (black/white/gray), darkness is part of the color,
        # so we don't penalize as much
        mean_value = np.mean(hsv[:, :, 2])
        if color in [Color.BLACK, Color.WHITE, Color.GRAY]:
            # Light penalty for grayscale detection in dim conditions
            if mean_value < 30:  # Very dark
                confidence_penalty = 0.7
            else:
                confidence_penalty = 1.0
        else:
            # Normal penalty for chromatic colors
            if mean_value < self.low_light_threshold:
                confidence_penalty = mean_value / self.low_light_threshold
            else:
                confidence_penalty = 1.0

        confidence = base_confidence * confidence_penalty

        if confidence < self.min_confidence:
            return Color.UNKNOWN, confidence

        return color, confidence

    def _classify_color_hsv(self, hsv: np.ndarray) -> tuple[Color, float]:
        """
        Classify dominant color from HSV image using binning.

        Returns:
            (color, confidence) where confidence is ratio of dominant color pixels
        """
        h, s, v = cv2.split(hsv)

        total_pixels = h.size
        if total_pixels == 0:
            return Color.UNKNOWN, 0.0

        # Handle grayscale (low saturation)
        low_sat_mask = s < self.GRAY_THRESHOLD_SAT
        gray_pixels = np.sum(low_sat_mask)

        if gray_pixels > 0.5 * total_pixels:
            # Check if black, white, or gray
            gray_values = v[low_sat_mask]
            mean_val = np.mean(gray_values)

            if mean_val < self.BLACK_THRESHOLD_VAL:
                confidence = min(0.9, gray_pixels / total_pixels)
                return Color.BLACK, confidence
            elif mean_val > self.WHITE_THRESHOLD_VAL:
                confidence = min(0.9, gray_pixels / total_pixels)
                return Color.WHITE, confidence
            else:
                confidence = min(0.9, gray_pixels / total_pixels)
                return Color.GRAY, confidence

        # Check for brown (special case: low-sat orange/yellow)
        brown_mask = (
            (h >= self.BROWN_HUE_RANGE[0])
            & (h <= self.BROWN_HUE_RANGE[1])
            & (s >= self.BROWN_SAT_RANGE[0])
            & (s <= self.BROWN_SAT_RANGE[1])
            & (v >= self.BROWN_VAL_RANGE[0])
            & (v <= self.BROWN_VAL_RANGE[1])
        )
        brown_pixels = np.sum(brown_mask)
        if brown_pixels > 0.3 * total_pixels:
            return Color.BROWN, brown_pixels / total_pixels

        # Check chromatic colors
        color_counts = {}

        for color, ranges in self.COLOR_RANGES.items():
            if len(ranges) == 4:  # Red wraps around
                lower1, upper1, lower2, upper2 = ranges
                mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = ranges
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            count = np.sum(mask > 0)
            if count > 0:
                color_counts[color] = count

        if not color_counts:
            return Color.UNKNOWN, 0.0

        # Return dominant color
        dominant_color = max(color_counts, key=color_counts.get)
        confidence = color_counts[dominant_color] / total_pixels

        return dominant_color, confidence

    def _has_bag(
        self,
        person_box: tuple[int, int, int, int],
        bag_boxes: list[tuple[int, int, int, int]] | None,
    ) -> bool:
        """
        Determine if person has a bag based on detector boxes.

        Args:
            person_box: (x1, y1, x2, y2)
            bag_boxes: List of (x1, y1, x2, y2) backpack/handbag boxes

        Returns:
            True if any bag box overlaps with person box (IoU > 0.1)
        """
        if not bag_boxes:
            return False

        px1, py1, px2, py2 = person_box
        person_area = (px2 - px1) * (py2 - py1)

        if person_area == 0:
            return False

        for bx1, by1, bx2, by2 in bag_boxes:
            # Compute intersection
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (ix2 - ix1) * (iy2 - iy1)
            bag_area = (bx2 - bx1) * (by2 - by1)

            if bag_area == 0:
                continue

            # IoU relative to bag (bag must be mostly inside person)
            iou = intersection / bag_area

            if iou > 0.1:
                return True

        return False


class AttributeQuery:
    """Helper for attribute-based search queries."""

    @staticmethod
    def matches(attrs: PersonAttributes, query: dict) -> bool:
        """
        Check if attributes match a query.

        Query format:
            {
                "upper_color": "red",  # optional
                "lower_color": "blue",  # optional
                "has_bag": True,  # optional
                "min_confidence": 0.5,  # optional, applies to both colors
            }

        Returns:
            True if all specified query conditions match
        """
        min_conf = query.get("min_confidence", 0.0)

        # Check confidence threshold
        if "upper_color" in query and attrs.upper_confidence < min_conf:
            return False
        if "lower_color" in query and attrs.lower_confidence < min_conf:
            return False

        # Check upper color
        if "upper_color" in query:
            if attrs.upper_clothing_color != Color(query["upper_color"]):
                return False

        # Check lower color
        if "lower_color" in query:
            if attrs.lower_clothing_color != Color(query["lower_color"]):
                return False

        # Check bag
        if "has_bag" in query:
            if attrs.has_bag != query["has_bag"]:
                return False

        return True
