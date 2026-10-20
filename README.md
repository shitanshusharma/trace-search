# trace-search

**Anonymous same-day cross-camera person route reconstruction for forensic search**

## Status: CODE-ONLY PROTOTYPE

This repository contains **code only**. No real footage, no consented footage, no trained model weights are included yet. This is task T0 of the Cut 01 build: the scaffold and license fence.

## What is trace-search?

trace-search is an on-premises, anonymous, same-day cross-camera person route reconstruction and forensic search prototype designed for mixed-brand CCTV video (Indian hospitals and tech parks).

**What it does:**
- Tracks one anonymous person's route across multiple cameras on the same day
- Works with mixed Hikvision/CP Plus estates (2+ brands, 6+ cameras)
- Ingests video files from NVR native exports (optional RTSP)
- Person detection, single-camera tracking, cross-camera re-identification (ReID to be trained on consented data)
- Attribute-based search: clothing color, bags/backpacks
- Web UI with site map, route visualization, and clip playback
- Audit logging and configurable retention

**What it does NOT do (by design):**
- ❌ No face recognition
- ❌ No gait analysis
- ❌ No watchlists or named identities
- ❌ No gender, age, or ethnicity outputs
- ❌ No cross-day or clothing-change matching
- ❌ No real-time alerts (human review mode only)

This is investigative tooling with human review, not automated surveillance.

## License Compliance

This project maintains **strict license hygiene** for commercial use:

- **Banned licenses:** AGPL, GPL, NC (Non-Commercial), unknown/unrecognized
- **GCC Runtime Library Exception:** The single GPL-family exception is GCC runtime libraries (`libgfortran`, `libquadmath`, `libgcc_s`, `libstdc++`) with the GCC Runtime Library Exception (`GPL-3.0 WITH GCC-exception-3.1`), which are required runtime components
- **Banned components:** Ultralytics (AGPL), boxmot (AGPL), InsightFace/ArcFace, OpenGait
- **Banned datasets:** Market-1501, MSMT17, DukeMTMC, CUHK03, CASIA-B, GREW, Gait3D, Objects365-trained weights

Every dependency, model, weight, dataset, and test clip is tracked in [`licenses/manifest.json`](licenses/manifest.json) with:
- Name, version, license, source URL, SHA-256
- For video clips: license version, date checked (IST), perceptual hashes, usage tag

CI verifies the full installed closure (direct + transitive dependencies) against the manifest and bans GPL/AGPL/SSPL/BUSL/NC/proprietary licenses and problematic components.

## Project Structure

```
trace-search/
├── src/trace_search/       # Main package
│   ├── ingest/             # Video file/RTSP ingest
│   ├── detection/          # Person detection (YOLOX-based)
│   ├── tracking/           # Single-camera tracking (OC-SORT/ByteTrack)
│   ├── attributes/         # Clothing color + bag detection
│   ├── reid/               # Cross-camera ReID (OSNet, trained on consented data)
│   ├── matching/           # Route reconstruction with topology constraints
│   ├── api/                # Search and query API
│   ├── ui/                 # Web UI (site map, route view, clips)
│   └── common/             # Shared utilities
├── tests/                  # Unit and integration tests
├── scripts/                # Build and maintenance scripts
│   └── check_licenses.py   # License fence validator
├── licenses/               # License manifest and schema
│   ├── manifest.json       # Machine-readable license manifest
│   └── manifest_schema.json
└── .github/workflows/      # CI/CD
```

## Installation

### Requirements

- Python 3.10+
- Ubuntu 20.04+ or similar Linux (on-prem deployment)
- GPU: NVIDIA RTX 4070-class (12 GB) or equivalent for production
- CPU: 8+ cores
- RAM: 32 GB

### Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd trace-search

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linter
ruff check src/ tests/ scripts/

# Check license manifest
python3 scripts/check_licenses.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/trace_search --cov-report=html

# Run only license fence tests
pytest tests/test_license_fence.py -v

# Test the license checker directly
python3 scripts/check_licenses.py
```

## CI Status

The CI pipeline runs on every push and PR:
1. **Lint** with ruff
2. **Unit tests** with pytest
3. **License fence check** - fails on any AGPL/GPL/NC/unknown license or banned component

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details.

## Development Workflow

This is a **multi-task build** (T0–T10). Each task = one cloud agent / one PR. Current status:

- ✅ **T0 (this PR):** Scaffold + license fence
- ⏳ **T1:** Synthetic smoke fixtures + SMOKE_ONLY store
- ⏳ **T2:** Ingest (video file + optional RTSP)
- ⏳ **T3:** Detection + tracking (YOLOX + OC-SORT)
- ⏳ **T4:** Attributes (color + bag)
- ⏳ **T5:** ReID (OSNet, train from scratch)
- ⏳ **T6:** Cross-camera matching + route
- ⏳ **T7:** Web UI + auth
- ⏳ **T8:** Audit log
- ⏳ **T9:** Retention + purge
- ⏳ **T10:** Zero-face guarantee

See [trace-search-cut01-plan.md](docs/trace-search-cut01-plan.md) for the full task breakdown.

## Privacy and Ethics

- **No face recognition.** No face embeddings, no face crops, no identity lookup.
- **No demographic outputs.** No gender, age, ethnicity, or other sensitive attributes.
- **Human review only.** Every result is reviewed by a human operator before action.
- **Short retention.** Footage-derived data purged after 7 days (default). Audit logs kept for 1 year.
- **Consent-first.** All training and evaluation use consented, self-recorded fixtures. No public datasets.
- **Anonymous queries.** No names, no watchlists. Queries require a case ID and purpose (audit logged).

This tool is designed for **investigative forensic search** (e.g., tracing a lost patient or incident review), not real-time surveillance or automated alerts.

## Legal and Compliance

- **India DPDP Act 2023:** Designed for private-site legal basis (Sec 7(a) with signage). Consent Manager registration and full duties start 13 May 2027.
- **No EU export yet.** Body-appearance tracking counts as biometric data under GDPR. EU deployment blocked pending legal review.
- **Consented data only.** All training, evaluation, and demos use consented, self-recorded fixtures. No real buyer footage until Cut 02 (paid pilot under NDA).

See the PRD for full risk assessment and legal basis.

## Contributing

This is a **locked-scope prototype** for Cut 01. External contributions are not accepted yet.

Internal contributors: follow the task plan. Each task gets one PR. Run tests and license checks before pushing.

## License

MIT License. See [LICENSE](LICENSE) for full text.

All dependencies are tracked in [`licenses/manifest.json`](licenses/manifest.json). CI verifies that all installed packages (direct + transitive) have allowed licenses and are not on the banned list. See AC-01 for details.

## Contact

Owner: Shitanshu Sharma  
Cut 01 target date: Fri 4 Dec 2026 (IST)  
Kill/continue review: Tue 8 Dec 2026 (IST)
