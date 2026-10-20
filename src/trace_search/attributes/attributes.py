"""
Deterministic person attribute extraction (T4).

Extracts upper-clothing color, lower-clothing color, and bag presence
from person bounding boxes. No trained weights, no sensitive attributes.
"""

from dataclasses import dataclass
from enum import Enum

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
    DARK_BLUE = "dark_blue"  # Navy, denim
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    UNKNOWN = "unknown"
    INVALID_BOX = "invalid_box"  # Degenerate or out-of-frame box


@dataclass(frozen=True)
class PersonAttributes:
    """
    Track appearance attributes (anonymous, no identity).

    Contains ONLY non-sensitive attributes:
    - upper_clothing_color
    - lower_clothing_color
    - bag_on_track
    - upper_confidence
    - lower_confidence

    Explicitly excludes: gender, age, ethnicity, face features, height, pose, identity.
    """

    upper_clothing_color: Color
    lower_clothing_color: Color
    bag_on_track: bool
    upper_confidence: float
    lower_confidence: float

    def __post_init__(self):
        if not (0.0 <= self.upper_confidence <= 1.0):
            raise ValueError(f"upper_confidence must be in [0,1], got {self.upper_confidence}")
        if not (0.0 <= self.lower_confidence <= 1.0):
            raise ValueError(f"lower_confidence must be in [0,1], got {self.lower_confidence}")


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

    # HSV color boundaries (H: 0-180, S: 0-255, V: 0-255, same scale as OpenCV)
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
        Color.BLUE: [(101, 100, 130), (130, 255, 255)],  # Bright blue
        Color.DARK_BLUE: [(101, 50, 50), (130, 255, 129)],  # Navy, denim (lower V than bright blue)
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
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0,1], got {min_confidence}")
        if not (0 <= low_light_threshold <= 255):
            raise ValueError(f"low_light_threshold must be in [0,255], got {low_light_threshold}")
        self.min_confidence = min_confidence
        self.low_light_threshold = low_light_threshold

    @staticmethod
    def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to HSV using numpy only (no OpenCV).

        Args:
            bgr: BGR image, shape (H, W, 3), dtype uint8

        Returns:
            HSV image with same H/S/V scale as OpenCV:
            - H: [0, 180] (degrees / 2)
            - S: [0, 255]
            - V: [0, 255]
        """
        bgr_normalized = bgr.astype(np.float32) / 255.0
        b, g, r = bgr_normalized[:, :, 0], bgr_normalized[:, :, 1], bgr_normalized[:, :, 2]

        v = np.maximum(np.maximum(r, g), b)
        c_max_min_diff = v - np.minimum(np.minimum(r, g), b)

        # Saturation (avoid divide by zero)
        s = np.where(v > 0, c_max_min_diff / v, 0.0)

        # Hue
        h = np.zeros_like(v)
        mask = c_max_min_diff != 0

        # Red is max
        r_max = (r == v) & mask
        h[r_max] = 60 * (((g[r_max] - b[r_max]) / c_max_min_diff[r_max]) % 6)

        # Green is max
        g_max = (g == v) & mask
        h[g_max] = 60 * (((b[g_max] - r[g_max]) / c_max_min_diff[g_max]) + 2)

        # Blue is max
        b_max = (b == v) & mask
        h[b_max] = 60 * (((r[b_max] - g[b_max]) / c_max_min_diff[b_max]) + 4)

        # Scale to OpenCV range: H [0,180], S [0,255], V [0,255]
        h = (h / 2).astype(np.uint8)
        s = (s * 255).astype(np.uint8)
        v = (v * 255).astype(np.uint8)

        return np.stack([h, s, v], axis=-1)

    def extract(
        self,
        frame: np.ndarray,
        person_box: tuple[int, int, int, int] | tuple[float, float, float, float],
        bag_boxes: list[tuple[int, int, int, int] | tuple[float, float, float, float]] | None = None,
    ) -> PersonAttributes:
        """
        Extract attributes from a person bounding box.

        Args:
            frame: BGR image (H, W, 3), dtype uint8
            person_box: (x1, y1, x2, y2) in pixel coordinates (int or float)
            bag_boxes: Optional list of (x1, y1, x2, y2) backpack/handbag boxes (int or float)

        Returns:
            PersonAttributes with color and bag information

        Raises:
            ValueError: If frame is invalid format
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be (H,W,3), got shape {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {frame.dtype}")

        # Clip box to frame boundaries and round floats to int
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        x1_clipped = int(round(max(0, min(x1, w))))
        y1_clipped = int(round(max(0, min(y1, h))))
        x2_clipped = int(round(max(0, min(x2, w))))
        y2_clipped = int(round(max(0, min(y2, h))))

        # Check for degenerate box (zero area or invalid coordinates)
        if x1_clipped >= x2_clipped or y1_clipped >= y2_clipped:
            return PersonAttributes(
                upper_clothing_color=Color.INVALID_BOX,
                lower_clothing_color=Color.INVALID_BOX,
                bag_on_track=False,
                upper_confidence=0.0,
                lower_confidence=0.0,
            )

        person_crop = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

        if person_crop.size == 0:
            return PersonAttributes(
                upper_clothing_color=Color.INVALID_BOX,
                lower_clothing_color=Color.INVALID_BOX,
                bag_on_track=False,
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

        # Clip bag boxes and determine bag presence
        clipped_bag_boxes = self._clip_boxes(bag_boxes, w, h) if bag_boxes else None
        bag_on_track = self._bag_on_track(
            (x1_clipped, y1_clipped, x2_clipped, y2_clipped), clipped_bag_boxes
        )

        return PersonAttributes(
            upper_clothing_color=upper_color,
            lower_clothing_color=lower_color,
            bag_on_track=bag_on_track,
            upper_confidence=upper_conf,
            lower_confidence=lower_conf,
        )

    @staticmethod
    def _clip_boxes(
        boxes: list[tuple[int, int, int, int] | tuple[float, float, float, float]],
        width: int,
        height: int,
    ) -> list[tuple[int, int, int, int]]:
        """Clip boxes to frame boundaries, round floats, and remove degenerate boxes."""
        clipped = []
        for x1, y1, x2, y2 in boxes:
            x1_c = int(round(max(0, min(x1, width))))
            y1_c = int(round(max(0, min(y1, height))))
            x2_c = int(round(max(0, min(x2, width))))
            y2_c = int(round(max(0, min(y2, height))))
            if x1_c < x2_c and y1_c < y2_c:  # Valid box
                clipped.append((x1_c, y1_c, x2_c, y2_c))
        return clipped

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

        if h < 40 or w < 20:  # Minimum reasonable person size
            return Color.UNKNOWN, 0.0

        # Define regions with margin trimming to exclude background
        # Trim 10% horizontal margin on each side, 5% vertical margin top/bottom of region
        margin_x = max(1, int(0.1 * w))
        margin_y = max(1, int(0.05 * h))

        if region == "upper":
            # Torso: 25% to 55% of height (narrower, avoiding head/waist)
            y_start = int(0.25 * h) + margin_y
            y_end = int(0.55 * h) - margin_y
        else:  # lower
            # Legs: 60% to 90% of height (avoiding feet region at bottom)
            y_start = int(0.60 * h) + margin_y
            y_end = int(0.90 * h) - margin_y

        y_start = max(0, y_start)
        y_end = min(h, y_end)

        if y_start >= y_end:
            return Color.UNKNOWN, 0.0

        # Trim horizontal margins
        x_start = margin_x
        x_end = w - margin_x

        if x_start >= x_end:
            return Color.UNKNOWN, 0.0

        region_crop = person_crop[y_start:y_end, x_start:x_end]

        if region_crop.size == 0:
            return Color.UNKNOWN, 0.0

        # Convert to HSV
        hsv = self.bgr_to_hsv(region_crop)

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
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

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
                mask1 = (
                    (h >= lower1[0])
                    & (h <= upper1[0])
                    & (s >= lower1[1])
                    & (s <= upper1[1])
                    & (v >= lower1[2])
                    & (v <= upper1[2])
                )
                mask2 = (
                    (h >= lower2[0])
                    & (h <= upper2[0])
                    & (s >= lower2[1])
                    & (s <= upper2[1])
                    & (v >= lower2[2])
                    & (v <= upper2[2])
                )
                mask = mask1 | mask2
            else:
                lower, upper = ranges
                mask = (
                    (h >= lower[0])
                    & (h <= upper[0])
                    & (s >= lower[1])
                    & (s <= upper[1])
                    & (v >= lower[2])
                    & (v <= upper[2])
                )

            count = np.sum(mask)
            if count > 0:
                color_counts[color] = count

        if not color_counts:
            return Color.UNKNOWN, 0.0

        # Return dominant color
        dominant_color = max(color_counts, key=color_counts.get)
        confidence = color_counts[dominant_color] / total_pixels

        return dominant_color, confidence

    def _bag_on_track(
        self,
        person_box: tuple[int, int, int, int],
        bag_boxes: list[tuple[int, int, int, int]] | None,
    ) -> bool:
        """
        Determine if track has a bag based on detector boxes.

        Uses intersection over BAG area (containment ratio).
        Frame-level one-to-one assignment: each bag goes to at most one track.

        Args:
            person_box: (x1, y1, x2, y2)
            bag_boxes: List of (x1, y1, x2, y2) backpack/handbag boxes

        Returns:
            True if any bag is assigned to this track via greedy containment
        """
        if not bag_boxes:
            return False

        px1, py1, px2, py2 = person_box
        person_area = (px2 - px1) * (py2 - py1)

        if person_area == 0:
            return False

        # Compute containment ratio for each bag (intersection / bag_area)
        containments = []
        for bx1, by1, bx2, by2 in bag_boxes:
            # Compute intersection
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)

            if ix2 <= ix1 or iy2 <= iy1:
                containments.append(0.0)
                continue

            intersection = (ix2 - ix1) * (iy2 - iy1)
            bag_area = (bx2 - bx1) * (by2 - by1)

            if bag_area == 0:
                containments.append(0.0)
                continue

            # Containment = intersection / bag_area (how much of bag is inside person)
            containment = intersection / bag_area
            containments.append(containment)

        # Find best bag for this person (greedy one-to-one assignment)
        best_containment = max(containments) if containments else 0.0

        # Assign bag if containment > threshold (0.5 = at least half the bag inside)
        return best_containment > 0.5


class AttributeQuery:
    """Helper for attribute-based search queries on anonymous tracks."""

    @staticmethod
    def matches(attrs: PersonAttributes, query: dict) -> bool:
        """
        Check if attributes match a query.

        Query format:
            {
                "upper_color": "red",  # optional
                "lower_color": "blue",  # optional
                "bag_on_track": True,  # optional
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
        if "bag_on_track" in query:
            if attrs.bag_on_track != query["bag_on_track"]:
                return False

        return True
