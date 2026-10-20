# trace-search

Anonymous cross-camera person route search prototype (Cut 01, T4: Attribute Extraction).

## Overview

This module implements deterministic person attribute extraction for privacy-preserving cross-camera tracking:

- **Upper-clothing color**: HSV-based classification into a 12-color palette
- **Lower-clothing color**: Separate classification for leg regions
- **Bag presence**: Derived from YOLOX backpack/handbag detections
- **No sensitive attributes**: No gender, age, ethnicity, face, height, or pose

## Features

### Deterministic Color Classification

Uses HSV color space binning with:
- 12-color palette: black, white, gray, red, orange, yellow, green, cyan, blue, purple, pink, brown
- Confidence scoring (returns "unknown" when uncertain)
- Low-light robustness via value channel analysis
- Region-based extraction (torso: 20-60% of person height, legs: 60-100%)

### Bag Detection

- Associates COCO `backpack`/`handbag` boxes with person boxes via IoU
- No learned weights required

### Query Interface

```python
from trace_search.attributes import AttributeExtractor, AttributeQuery

extractor = AttributeExtractor()

# Extract from frame + person box
attrs = extractor.extract(
    frame=frame,  # BGR image
    person_box=(x1, y1, x2, y2),
    bag_boxes=[(bx1, by1, bx2, by2), ...],  # optional
)

# Query
query = {"upper_color": "red", "has_bag": True}
matches = AttributeQuery.matches(attrs, query)
```

## Color Palette

| Color | HSV Range | Notes |
|-------|-----------|-------|
| Red | H: 0-10, 170-180 | Wraps around hue circle |
| Orange | H: 11-25 | |
| Yellow | H: 26-35 | |
| Green | H: 36-85 | |
| Cyan | H: 86-100 | |
| Blue | H: 101-130 | |
| Purple | H: 131-160 | |
| Pink | H: 161-169 | |
| Brown | H: 11-35, low saturation | Special case |
| Black | V < 50 | |
| White | V > 200 | |
| Gray | S < 40, 50 < V < 200 | |

## AC-08 Harness

Precision@10 evaluation for attribute queries:

```python
from trace_search.ac08_harness import AC08Harness, LabelledRecord, AttributeSearchQuery

harness = AC08Harness()
harness.add_record(LabelledRecord("person_001", attrs))
# ... add more records

queries = [
    AttributeSearchQuery("q1", {"upper_color": "red"}, ["person_001", "person_005"]),
]

results = harness.evaluate(queries)
print(results["mean_precision_at_10"])  # Target: >= 0.80 for AC-08
```

## Known Limitations

1. **Occlusion**: Partial occlusion reduces confidence but may still misclassify
2. **Low light**: Very dim scenes (mean V < 80) return reduced confidence or unknown
3. **Mixed patterns**: Highly patterned clothing may return lower confidence or incorrect color
4. **Small boxes**: Boxes < 20px height return unknown
5. **Color ambiguity**: Brown vs. dark orange, pink vs. light red can be confused at boundaries
6. **Bag association**: Simple IoU thresholding; may miss carried bags outside person box

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=trace_search
```

Tests cover:
- ✓ All 12 palette colors on synthetic patches
- ✓ Synthetic person crops with upper/lower regions
- ✓ Unknown path for low confidence
- ✓ Dim-light behavior
- ✓ Bag detection with overlapping boxes
- ✓ Schema validation (no sensitive attributes)
- ✓ AC-08 precision@10 harness

## License

MIT License - see LICENSE file.

## Privacy & Ethics

This implementation explicitly excludes:
- ❌ Gender, age, ethnicity detection
- ❌ Face recognition or face embeddings
- ❌ Identity, name, or watchlist features
- ❌ Height, pose, or gait analysis
- ❌ Any trained attribute models (PA-100K, RAP, PP-Human)

Per Cut 01 spec (AC-12), a test asserts the attribute schema contains none of these fields.

## Input Contract

Expected from upstream detector (T3, YOLOX):

```python
{
    "frame": np.ndarray,  # BGR image, shape (H, W, 3)
    "person_box": (x1, y1, x2, y2),  # ints, absolute pixel coords
    "bag_boxes": [(x1, y1, x2, y2), ...],  # optional, COCO backpack/handbag
}
```

## Output Schema

```python
@dataclass
class PersonAttributes:
    upper_clothing_color: Color  # enum
    lower_clothing_color: Color
    has_bag: bool
    upper_confidence: float  # [0, 1]
    lower_confidence: float
```

## Development Status

**T4 (Attribute Extraction)** - Ready for integration.

Other tasks (T0-T3, T5-T10) are handled by separate agents per the plan.

---

**Repo**: `shitanshusharma/trace-search` (public)  
**Owner**: Shitanshu Sharma  
**Cut 01 Target**: Fri 4 Dec 2026 (code-only mode)
