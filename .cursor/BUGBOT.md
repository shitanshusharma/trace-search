# Bugbot Configuration for trace-search

This file defines the hard-stops and critical constraints from the PRD that Bugbot should enforce when reviewing code changes.

## PRD Hard-Stops (Cut 01 rev 2)

### 1. Banned Dependencies & Licenses

**FAIL immediately if any PR adds:**
- AGPL, GPL (any version), SSPL, BUSL, CC-BY-SA, or NC (Non-Commercial) licenses
- Research-only or academic-only dependencies
- Ultralytics (AGPL-3.0) or boxmot (AGPL)
- Any unknown or unrecognized license

**Banned specific dependencies:**
- face_recognition, insightface, arcface, retinaface, dlib face models
- deepface, facenet, mediapipe face detection/landmarks
- opengait, OpenGait
- ultralytics (YOLO), boxmot

**Banned datasets/weights:**
- Market-1501, MSMT17, DukeMTMC, CUHK03 ReID datasets
- CASIA-B, GREW, Gait3D gait datasets
- Objects365-trained weights
- ImageNet-pretrained backbones (where avoidable)
- LAION, DataComp, CLIP weights trained on those
- Public ReID model-zoo weights (Market/Duke-trained)
- Any research-only weights without explicit commercial license

### 2. No Face Recognition / Biometric Identity Features

**FAIL if any PR introduces:**
- Face recognition, face embeddings, or face crops
- Face detection models or face landmark detection
- Gait recognition or gait features
- Watchlists, enrolment databases, or named-identity lookup
- Identity matching across days (cross-day ReID)
- Name fields, national_id fields, or person identity storage
- Gender, age, ethnicity, or other demographic attribute outputs

**Code patterns to reject:**
- Imports: `face_recognition`, `insightface`, `arcface`, `retinaface`, `deepface`, `facenet`, `mediapipe.solutions.face_*`, `opengait`
- Identifiers: `watchlist`, `enroll`, `enrolment`, `face_embedding`, `face_crop`, `gait_`, `national_id`, `identity_id`, `person_name`
- API schema fields: `name`, `national_id`, `identity`, `gender`, `age`, `ethnicity`
- Face data storage: embeddings labeled as "face", crops from face regions

### 3. SMOKE_ONLY Separation (AC-02b)

**FAIL if:**
- Any fixture tagged `SMOKE_ONLY` or `CODE_TEST` appears in training, eval, FX-HOLD, or study inputs
- Manifest rows with `use=SMOKE_ONLY` or `use=CODE_TEST` referenced in training configs
- Stock footage (Pexels/Pixabay) or synthetic smoke video used for anything except detection/tracking bring-up

**In code-only mode (until unparked):**
- No training jobs of any kind on any footage (stock, synthetic, or consented)
- ReID must remain an untrained placeholder
- No public or research-only ReID weights loaded

### 4. On-Premises Only / No Cloud / No Outbound Data

**FAIL if any PR adds:**
- Cloud API calls (AWS, Azure, GCP, external services)
- Outbound network requests (except localhost/LAN)
- Telemetry, analytics, or phone-home code
- Cloud-hosted model inference APIs
- Any data exfiltration or external logging services

**Allowed:**
- On-prem deployment only
- Local RTSP streams from NVRs
- LAN-only web UI access

### 5. Media Files in Repo

**FAIL if media files committed outside approved paths:**
- Video: .mp4, .mkv, .avi, .mov, .h264, .h265, .dav, .webm
- Images: .jpg, .jpeg, .png (except docs diagrams in `docs/`)
- Weights: .pt, .pth, .onnx, .safetensors, .h5, .pb

**Allowed paths:**
- `fixtures/`
- `tests/fixtures/`
- `docs/diagrams/` (diagrams only)

All media must be in the license manifest with SHA-256 and (for video) pHash.

### 6. License Manifest Integrity (AC-01)

**FAIL if:**
- Any dependency is not listed in `licenses/manifest.json`
- Manifest entry is missing: name, version, license, source_url, SHA-256
- Video clips missing: license_version, date_checked, pHash, `use` tag
- Scanner report shows AGPL/GPL/NC/unknown results
- FFmpeg is not LGPL-only build (no `--enable-gpl` or `--enable-nonfree`)

### 7. Retention & Privacy (AC-12 to AC-15)

**FAIL if any PR:**
- Adds face embedding storage
- Adds face crop persistence
- Stores whole-body crops beyond retention TTL
- Adds name/identity fields to any schema
- Creates public share links (not in scope for Cut 01)
- Bypasses audit logging for search/view/export actions
- Allows footage-derived data to persist beyond configured TTL

**Required:**
- Audit log: who, when, action, case_id, purpose, result_ids
- Audit log retention: 1 year (independent of footage TTL)
- Footage-derived data TTL: 7 days default, configurable
- Access control: 401 for all unauthenticated search/route/clip/export requests

### 8. No Public Datasets for Training/Eval

**FAIL if any PR uses these for training or evaluation:**
- Market-1501, MSMT17, DukeMTMC, CUHK03 (ReID)
- CASIA-B, GREW, Gait3D (gait)
- Objects365 (detection)
- PA-100K, RAP (attributes)
- LAION, DataComp, CLIP (embeddings)

**Allowed for training/eval:**
- Consented, self-recorded fixtures only (FX-TRAIN, FX-DEV, FX-HOLD)
- Synthetic code-test fixtures (FX-CODE) for functional tests only
- Stock clips (Pexels/Pixabay) for smoke test bring-up only (AC-02b separation enforced)

### 9. Public Webcams & Unsecured Cameras (§2 OUT)

**FAIL if any PR:**
- Adds ingest from public webcam streams (EarthCam, YouTube Live, etc.)
- Accesses unsecured-camera aggregators (Insecam, Shodan camera search, etc.)
- Scrapes or bulk-downloads from stock footage sites

### 10. Code-Only Mode Constraints (Until Unparked)

**Current state: CODE-ONLY prototype (§0)**

**FAIL if any PR:**
- Adds training code that runs on any footage (stock, synthetic, or consented)
- Loads pre-trained ReID weights (must stay random-init placeholder)
- Reports any accuracy metric as meaningful (placeholder ReID outputs are random)
- Uses FX-SMOKE or FX-CODE clips for accuracy scoring or tuning
- Claims to prove AC-05, AC-06, AC-07, AC-08, or AC-11 (all DEFERRED)

**Allowed:**
- Untrained ReID placeholder (random-init OSNet or stub embedder)
- Functional/latency/privacy ACs on FX-CODE (AC-09, AC-10, AC-12–AC-15)
- Smoke test on stock/synthetic (detection/tracking bring-up only, no scoring)

---

## Enforcement Notes

- **These are BLOCKING failures.** Any violation should result in an immediate FAIL recommendation.
- **No soft-pass.** "Mostly compliant", "will fix later", "only for testing" are not acceptable.
- **Allow-list for false positives:** `.project-rules-allow` exists for documented exceptions. Each entry must have a reviewed justification comment.
- **Counsel review is PARKED** in code-only mode, but license hygiene is still enforced.
- **Milestone M8 (Tue 8 Dec 2026)** is a pipeline-ready review, not a kill/continue review until parked items are unparked.

## Review Checklist

When reviewing a PR, check:

1. ✅ No AGPL/GPL/NC/unknown licenses added
2. ✅ No banned dependencies (face recognition, gait, Ultralytics, etc.)
3. ✅ No face/gait/watchlist/identity code or schema fields
4. ✅ No training code on any footage in code-only mode
5. ✅ No public/research ReID or face datasets referenced
6. ✅ No SMOKE_ONLY or CODE_TEST clips in train/eval paths
7. ✅ No media files committed outside allowed paths
8. ✅ All new dependencies in license manifest with clean licenses
9. ✅ No cloud APIs or outbound data calls
10. ✅ Audit logging and retention enforcement maintained

---

**Source:** PRD Cut 01 rev 2 (`cut-01-prototype_4f3a.md`), §0, §2, §2a, §3, §4, §5, §8  
**Owner:** Shitanshu Sharma  
**Last Updated:** Tue 20 Oct 2026 (IST)
