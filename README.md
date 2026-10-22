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
- **OpenCV bundled files (21 Oct 2026, 7:45 AM IST):** OpenSSL 1.1.1w (`OpenSSL`) on `libcrypto-337dac8b.so.1.1` and `libssl-28bef1ac.so.1.1`, and Intel IPP 2021.11.0 (`LicenseRef-Intel-Simplified-Software-License`) on `cv2/cv2.abi3.so`, only inside opencv-python-headless 4.10.0.84. Counsel confirmation required before release. These are not global allowlist entries
- **Pillow bundled FreeType:** FreeType 2.14.3 (`FTL`) on `pillow.libs/libfreetype-57f805ba.so.6.20.6`, only inside Pillow 13.0.0. This is not a global allowlist entry. The FreeType credit line is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- **Banned components:** Ultralytics (AGPL), boxmot (AGPL), InsightFace/ArcFace, OpenGait
- **Banned datasets:** Market-1501, MSMT17, DukeMTMC, CUHK03, CASIA-B, GREW, Gait3D, Objects365-trained weights

Every dependency, model, weight, dataset, and test clip is tracked in [`licenses/manifest.json`](licenses/manifest.json) with:
- Name, version, license, source URL, SHA-256
- For video clips: repo-relative `path`, license version, date checked (IST), perceptual hashes, usage tag

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
│   ├── fixtures/           # Synthetic fixture generation + separation guards (T1)
│   ├── audit/              # Append-only HMAC audit log (T8, AC-13)
│   └── common/             # Shared utilities
├── tests/                  # Unit and integration tests
├── scripts/                # Build and maintenance scripts
│   ├── check_licenses.py   # License fence validator
│   ├── generate_smoke_fixtures.py  # Synthetic video generator (T1)
│   ├── run_smoke.py                # M1a smoke runner (CI: scripted-figures)
│   ├── check_separation.py         # Fixture separation checker (T1)
│   └── demo_audit_log.py   # AC-13 JSON proof
├── licenses/               # License manifest and schema
│   ├── manifest.json       # Machine-readable license manifest
│   └── manifest_schema.json
├── smoke/                  # Synthetic smoke test fixtures (generated, not committed)
└── .github/workflows/      # CI/CD
```

## Installation

### Requirements

- Python 3.11 or 3.12 (Pillow 13 requires 3.11+)
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

# Run T1 smoke fixture tests
pytest tests/test_generator.py tests/test_guards_comprehensive.py tests/test_separation_comprehensive.py tests/test_t1_integration.py tests/test_manifest_pins.py -v

# Test the license checker directly
python3 scripts/check_licenses.py

# Generate smoke fixtures
python3 scripts/generate_smoke_fixtures.py --update-manifest

# Check separation (smoke vs train/dev/hold)
python3 scripts/check_separation.py --json
```

### Smoke runner (M1a)

The runner calls the camera pipeline with `derived_from='SMOKE'`. That output is not evidence for any acceptance criterion.

CI runs `scripted-figures` on the 5 synthetic clips only. It does not download weights, and it does not run `yolox` against real weights. `yolox` is local-only.

`scripted-figures` is an OpenCV threshold and contour pass over the synthetic generator's flat-colour figures. It labels every blob that clears the threshold as a person, including the generator's obstructions. The cutoffs are fitted to that generator. It is a SMOKE-wiring stub, not a real detector, and it is not run or tuned on stock clips.

```bash
# CI shape: 5 synthetic clips, no weights
python scripts/run_smoke.py \
  --manifest licenses/manifest.json \
  --clips-dir smoke \
  --db /tmp/smoke.sqlite \
  --detector scripted-figures \
  --min-clips 5
```

```bash
# Local only. Weights stay outside the repo. Set --weights or TRACE_SEARCH_YOLOX_WEIGHTS.
python scripts/run_smoke.py \
  --manifest /path/outside/the/repo/smoke-manifest.json \
  --clips-dir /path/outside/the/repo/clips \
  --db /tmp/smoke.sqlite \
  --detector yolox \
  --weights /path/outside/the/repo/yolox_s.onnx \
  --min-clips 5
```

Each clip is checked with the fixture guard before the pipeline opens it. The smoke guard allows an identified clip only when its tag is SMOKE_ONLY. Any other tag is refused, and so is a missing row. Each invocation prints its own `run_id` and counts only the `SMOKE` rows written under that id. The `tracks` number on each clip line is the distinct track ids stored for that run. The pipeline's own `tracks` field counts observation rows, and the runner does not print that field. `--min-clips` must be at least 1. A file in the clip directory that is not an `.mp4` is printed as skipped and is not counted. The process exits 0 only when at least `--min-clips` clips finish with 0 errors and each of those clips has at least one observation from this invocation.

The T-3-9 local real-weights `yolox` run on 5 Pexels/Pixabay clips (kept outside the repo, not in git) is recorded in the pull request from branch `t3-9-real-weights-run`. SMOKE: not evidence for any AC.


## CI Status

The CI pipeline runs on every push and PR:
1. **Lint** with ruff
2. **Unit tests** with pytest
3. **Clean worktree** — fails if `git status --porcelain` is not empty after pytest
4. **License fence check** - fails on any AGPL/GPL/NC/unknown license or banned component
5. **CI tools banned-licence check** — separate venv from `requirements-ci.lock`
6. **AC-13 demo** — `python scripts/demo_audit_log.py --json`
7. **Fixture separation** — separate job. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

numpy 1.26.4 and its manifest row come from #6. ImageHash 4.3.2 is recorded as BSD-2-Clause.

## Development Workflow

This is a **multi-task build** (T0–T10). Each task = one cloud agent / one PR. Current status:

- ✅ **T0:** Scaffold + license fence
- **T1:** Synthetic smoke fixtures and separation checks are in this branch. numpy 1.26.4 and its manifest row come from #6.
- ⏳ **T2:** Ingest (video file + optional RTSP)
- ⏳ **T3:** Detection + tracking (YOLOX + OC-SORT)
- ⏳ **T4:** Attributes (color + bag)
- ⏳ **T5:** ReID (OSNet, train from scratch)
- ⏳ **T6:** Cross-camera matching + route
- ⏳ **T7:** Web UI + auth
- ⏳ **T8:** Audit log (HMAC chain and signed head anchor; see below)
- ⏳ **T9:** Retention + purge
- ⏳ **T10:** Zero-face guarantee

See [trace-search-cut01-plan.md](docs/trace-search-cut01-plan.md) for the full task breakdown.

## Audit log (AC-13)

The audit log records operator username, IST time, action, query, case ID, purpose, an optional `details` note, and result ids. It is an HMAC-SHA256 chain. Canonical JSON is `json.dumps(..., sort_keys=True, separators=(",", ":"))`. A signed head anchor at `<log_path>.head` stores the last HMAC and the entry count. `initialize()` writes a signed count-0 anchor for a new log. `append()` checks that anchor under the lock and raises when it does not match the tail. A missing, unsigned, or mismatched anchor fails `verify()`. `verify()` takes a shared lock on the same lock file and returns `valid=False` for non-UTF-8 bytes and for a blank `details` string.

The key comes only from `TRACE_AUDIT_HMAC_KEY` or from an explicit key file outside the configured audit directory. There is no default key. `verify()` never creates a key. `init_hmac_key()` is the only function that creates one. `AUDIT_LOG_DIR` and `is_audit_path()` (exported from `trace_search.audit`) are the default path and the purge-exemption check. `is_audit_path()` matches the resolved configured directory only, not every folder named `audit_logs`.

An attacker who holds the key and deletes both the log and the anchor is not detectable on this host. An attacker without the key who restores an older copy of both the log and the anchor also passes `verify()`. Logs written before `details` existed fail HMAC verification, because the canonical payload now includes `"details": null`. External anchoring is out of scope for Cut 01. The 1-year purge checkpoint is T9. Audit retention is 365 days.

`tests/test_audit_log.py` is 55 tests, all under `tmp_path`. The full suite is 277 tests. Details and a sample proof are in [`src/trace_search/audit/README.md`](src/trace_search/audit/README.md).

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
